"""
OWL2Vec* recommender: given a SNOMED focus concept, rank candidate
attribute values using the trained embedding analogy:

    focus_vec + role_attr_vec  ≈  best_value_vec

This surfaces the most semantically plausible (attr, val) pairs for
building a post-coordinated expression from an ICD-10 mapping.
"""

import re
import sqlite3
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from gensim.models import Word2Vec

SNOMED_DB  = "/Users/remzicelebi/workspace/snomed/snomed_cache.db"
MODEL_PATH = "/Users/remzicelebi/workspace/snomed/snomed_owl2vec.model"

C  = "c:"
R  = "r:"


def _c(cid):  return C + str(cid)
def _r(aid):  return R + str(aid)

_STOPWORDS = {
    "and", "or", "of", "the", "a", "an", "in", "on", "at", "to",
    "for", "with", "due", "by", "as", "not", "other", "specified",
    "unspecified", "nos", "nec", "type", "types", "without",
}

def _tokenize(text: str) -> list[str]:
    """Clean label text to the same tokens used in lexical walks, minus stopwords."""
    tokens = re.sub(r"[^a-zA-Z0-9 ]", " ", text.lower()).split()
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


class OWL2VecRecommender:
    def __init__(self, model_path: str = MODEL_PATH, db_path: str = SNOMED_DB):
        self.model_path = model_path
        self.db_path    = db_path
        self._model: Optional[Word2Vec] = None
        self._ready     = threading.Event()
        self._lock      = threading.Lock()
        self._error: Optional[str] = None

    def is_ready(self) -> bool:
        return self._ready.is_set()

    def get_error(self) -> Optional[str]:
        return self._error

    def ensure_loaded(self):
        with self._lock:
            if self._ready.is_set():
                return
            if not Path(self.model_path).exists():
                self._error = (
                    f"OWL2Vec model not found at {self.model_path}. "
                    "Run: .venv311/bin/python owl2vec_train.py"
                )
                return
            try:
                print("[owl2vec] Loading model…")
                self._model = Word2Vec.load(self.model_path)
                print(f"[owl2vec] Ready — vocab: {len(self._model.wv):,}")
                self._ready.set()
            except Exception as e:
                self._error = str(e)

    # ── DB helpers ────────────────────────────────────────────────────────

    def _conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _label(self, cid: str) -> str:
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(pref_label, label, ?) FROM concepts WHERE id=?",
                (cid, cid),
            ).fetchone()
        return row[0] if row else cid

    def _role_values(self, attr_id: str) -> list[str]:
        with self._conn() as c:
            return [r[0] for r in c.execute(
                "SELECT DISTINCT val_id FROM roles WHERE attr_id=?", (attr_id,)
            )]

    # ── Core analogy ──────────────────────────────────────────────────────

    def rank_values(
        self,
        focus_snomed_code: str,
        attr_id: str,
        topn: int = 5,
    ) -> list[dict]:
        """
        Rank all ontologically valid values for (focus, attr) by
        cosine similarity of (focus_vec + role_attr_vec) to each value_vec.
        """
        if not self._ready.is_set() or self._model is None:
            return []

        wv         = self._model.wv
        focus_key  = _c(focus_snomed_code)
        attr_key   = _r(attr_id)

        if focus_key not in wv or attr_key not in wv:
            return []

        query = wv[focus_key] + wv[attr_key]
        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return []
        query = query / q_norm

        scored = []
        for val_id in self._role_values(attr_id):
            val_key = _c(val_id)
            if val_key not in wv:
                continue
            v = wv[val_key]
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                continue
            score = float(np.dot(query, v / v_norm))
            scored.append((score, val_id))

        scored.sort(reverse=True)
        return [
            {"id": vid, "label": self._label(vid), "score": round(s, 4)}
            for s, vid in scored[:topn]
        ]

    def enrich_post_coord(
        self,
        focus_snomed_code: str,
        post_coord: dict,
        topn: int = 5,
    ) -> dict:
        """
        Augments a post_coord dict (from SnomedOWL.suggest_post_coord) by
        adding OWL2Vec-ranked values to each attribute entry.
        Returns a new dict; does not mutate the original.
        """
        if not self._ready.is_set():
            return post_coord

        enriched = []
        for entry in post_coord.get("attributes", []):
            attr_id = entry["attr"]["id"]
            ranked  = self.rank_values(focus_snomed_code, attr_id, topn=topn)
            enriched.append({**entry, "values_owl2vec": ranked})

        # Also rank own_roles values (the defining roles of the concept)
        own_roles_enriched = []
        for role in post_coord.get("own_roles", []):
            attr_id = role["attr"]["id"]
            ranked  = self.rank_values(focus_snomed_code, attr_id, topn=topn)
            own_roles_enriched.append({**role, "values_owl2vec": ranked})

        return {
            **post_coord,
            "own_roles":  own_roles_enriched if own_roles_enriched else post_coord.get("own_roles", []),
            "attributes": enriched,
            "owl2vec_ready": True,
        }

    def most_similar_concept(self, snomed_code: str, topn: int = 10) -> list[dict]:
        """Return the topn most similar SNOMED concepts in embedding space."""
        if not self._ready.is_set() or self._model is None:
            return []
        key = _c(snomed_code)
        if key not in self._model.wv:
            return []
        results = self._model.wv.most_similar(key, topn=topn)
        out = []
        for token, score in results:
            if not token.startswith(C):
                continue
            cid = token[len(C):]
            out.append({"id": cid, "label": self._label(cid), "score": round(score, 4)})
        return out

    def match_by_label(self, icd10_label: str, topn: int = 10) -> list[dict]:
        """
        Embed an ICD-10 label as the mean of its token vectors (using the
        lexical walk vocabulary), then find the nearest SNOMED concepts.

        This is used when no Athena 'Maps to' relationship exists for the
        ICD-10 code — the label text itself drives the SNOMED match.
        """
        if not self._ready.is_set() or self._model is None:
            return []

        wv = self._model.wv
        tokens = _tokenize(icd10_label)
        vecs = [wv[t] for t in tokens if t in wv]
        if not vecs:
            return []

        query = np.mean(vecs, axis=0)
        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return []
        query = query / q_norm

        # Score all concept tokens (c: prefix) in the vocabulary
        # Build index lazily: concept_key → vector matrix
        if not hasattr(self, "_concept_keys"):
            self._concept_keys = [k for k in wv.key_to_index if k.startswith(C)]
            self._concept_matrix = np.vstack([wv[k] for k in self._concept_keys])
            norms = np.linalg.norm(self._concept_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self._concept_matrix_normed = self._concept_matrix / norms

        scores = self._concept_matrix_normed @ query
        top_idx = np.argpartition(scores, -topn)[-topn:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results = []
        for idx in top_idx:
            cid = self._concept_keys[idx][len(C):]
            results.append({
                "id":    cid,
                "label": self._label(cid),
                "score": round(float(scores[idx]), 4),
            })
        return results

    def suggest_post_coord_from_label(
        self, icd10_label: str, snomed_owl, topn_concepts: int = 5, topn_values: int = 3
    ) -> dict:
        candidates = self.match_by_label(icd10_label, topn=topn_concepts)
        if not candidates:
            return {"error": "No embedding match found"}
        results = []
        for cand in candidates:
            cid = cand["id"]
            pc  = snomed_owl.suggest_post_coord(cid)
            if pc:
                pc = self.enrich_post_coord(cid, pc, topn=topn_values)
            results.append({"candidate": cand, "post_coord": pc})
        return {"icd10_label": icd10_label, "method": "owl2vec", "matches": results}

    def suggest_post_coord_from_biobert(
        self,
        biobert_matches: list[dict],
        snomed_owl,
        topn_values: int = 5,
    ) -> dict:
        """
        Stage 2 of the two-stage pipeline:

        Given top-K SNOMED concepts retrieved by BioBERT (Stage 1):
          1. Collect each concept's OWL own-roles (defining post-coord pattern)
          2. Average their OWL2Vec vectors → consensus embedding
          3. For every attribute that appears in ≥1 concept's roles,
             predict the best value using: consensus_vec + role_attr_vec ≈ value
          4. Surface attributes common to multiple concepts (agreement score)

        Returns a consolidated post-coordinated expression suggestion.
        """
        if not self._ready.is_set() or self._model is None:
            return {}

        wv = self._model.wv

        # ── Per-concept roles and vectors ─────────────────────────────
        concept_data = []
        for m in biobert_matches:
            cid = m["id"]
            pc  = snomed_owl.suggest_post_coord(cid)
            vec = wv[_c(cid)] if _c(cid) in wv else None
            concept_data.append({"cid": cid, "label": m["label"],
                                  "score": m["score"], "pc": pc, "vec": vec})

        # ── Consensus vector (mean of concepts that have OWL2Vec embeddings) ──
        vecs = [d["vec"] for d in concept_data if d["vec"] is not None]
        if not vecs:
            return {}
        consensus     = np.mean(vecs, axis=0)
        consensus_normed = consensus / (np.linalg.norm(consensus) or 1)

        # ── Aggregate attributes across all top-K concepts ────────────
        # attr_id → {count, values_from_owl, owl2vec_ranked_values}
        attr_counts: dict[str, int]       = {}
        attr_owl_vals: dict[str, set]     = {}

        for d in concept_data:
            pc = d.get("pc") or {}
            for role in pc.get("own_roles", []):
                aid = role["attr"]["id"]
                vid = role["val"]["id"]
                attr_counts[aid] = attr_counts.get(aid, 0) + 1
                attr_owl_vals.setdefault(aid, set()).add(vid)

        # ── For each attribute, rank values using consensus analogy ───
        consensus_attrs = []
        for attr_id, count in sorted(attr_counts.items(), key=lambda x: -x[1]):
            attr_key = _r(attr_id)
            if attr_key not in wv:
                continue
            # Analogy: consensus + role_attr → predicted value
            query     = consensus_normed + wv[attr_key]
            q_norm    = np.linalg.norm(query)
            if q_norm == 0:
                continue
            query     = query / q_norm

            # Score candidate values (union of values from all top-K concepts)
            valid_vals = snomed_owl.get_role_values(attr_id)
            scored = []
            for val_id in valid_vals:
                val_key = _c(val_id)
                if val_key not in wv:
                    continue
                v = wv[val_key]
                v_norm = np.linalg.norm(v)
                if v_norm == 0:
                    continue
                scored.append((float(np.dot(query, v / v_norm)), val_id))
            scored.sort(reverse=True)

            top_vals = [
                {"id": vid, "label": snomed_owl.get_label(vid), "score": round(s, 4)}
                for s, vid in scored[:topn_values]
            ]
            # Mark values that appeared in the actual OWL roles of retrieved concepts
            owl_val_ids = attr_owl_vals.get(attr_id, set())
            for v in top_vals:
                v["in_owl"] = v["id"] in owl_val_ids

            consensus_attrs.append({
                "attr":      {"id": attr_id, "label": snomed_owl.get_label(attr_id)},
                "agreement": count,               # how many top-K concepts share this attr
                "topk":      len(biobert_matches),
                "values":    top_vals,
            })

        # ── Build suggested expression ────────────────────────────────
        top_concept = concept_data[0]
        expression  = _build_expression(top_concept, consensus_attrs)

        return {
            "method":           "biobert+owl2vec",
            "top_concepts":     [{"cid": d["cid"], "label": d["label"],
                                   "score": d["score"]} for d in concept_data],
            "consensus_attrs":  consensus_attrs,
            "expression":       expression,
        }


def _build_expression(top_concept: dict, consensus_attrs: list[dict]) -> str:
    """Format the suggested post-coordinated expression."""
    pc = top_concept.get("pc") or {}
    parents = pc.get("parents") or []
    focus_id    = top_concept["cid"]
    focus_label = top_concept["label"]
    fp = parents[0] if parents else {"id": focus_id, "label": focus_label}

    lines = [f"{fp['id']} | {fp['label']} | :"]
    lines.append("  {")
    for a in consensus_attrs:
        if not a["values"]:
            continue
        best = a["values"][0]
        mark = "  [OWL✓]" if best.get("in_owl") else ""
        lines.append(
            f"    {a['attr']['id']} | {a['attr']['label']} |"
            f" = {best['id']} | {best['label']} |"
            f"  // agreement {a['agreement']}/{a['topk']}, sim {best['score']}{mark}"
        )
    lines.append("  }")
    return "\n".join(lines)
