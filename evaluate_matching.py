"""
Evaluation: how often do vector-based top-K recommendations contain
the correct SNOMED concept from the Athena direct mapping?

Metrics:
  Hit@K  — direct mapping SNOMED code appears in top-K recommendations
  MRR    — mean reciprocal rank of the correct concept
  Avg sim — average cosine similarity of the correct concept when found

Usage:
    .venv311/bin/python evaluate_matching.py [--sample N] [--method biobert|owl2vec|both]
"""

import argparse
import random
import sqlite3
import re
import sys
import time
import numpy as np
from collections import defaultdict

ATHENA_DB = "/Users/remzicelebi/workspace/snomed/athena_vocab_cache.db"
SNOMED_DB = "/Users/remzicelebi/workspace/snomed/snomed_cache.db"
OWL2VEC_MODEL = "/Users/remzicelebi/workspace/snomed/snomed_owl2vec.model"
BIOBERT_IDS   = "/Users/remzicelebi/workspace/snomed/biobert_ids.npy"
BIOBERT_VECS  = "/Users/remzicelebi/workspace/snomed/biobert_vecs.npy"

STOPWORDS = {
    "and","or","of","the","a","an","in","on","at","to",
    "for","with","due","by","as","not","other","specified",
    "unspecified","nos","nec","type","types","without",
}


# ── Data loading ───────────────────────────────────────────────────────────

def load_test_pairs(sample_n: int) -> list[dict]:
    """Load ICD-10 → SNOMED direct mapping pairs from Athena."""
    conn = sqlite3.connect(ATHENA_DB)
    rows = conn.execute("""
        SELECT
            icd.concept_code   AS icd_code,
            icd.concept_name   AS icd_label,
            sno.concept_code   AS snomed_code,
            sno.concept_name   AS snomed_label
        FROM concepts icd
        JOIN relationships r ON r.concept_id_1 = icd.concept_id
        JOIN concepts sno    ON sno.concept_id = r.concept_id_2
        WHERE icd.vocabulary_id = 'ICD10CM'
          AND sno.vocabulary_id = 'SNOMED'
          AND sno.standard = 'S'
          AND r.relationship_id = 'Maps to'
          AND (icd.invalid_reason IS NULL OR icd.invalid_reason = '')
          AND (sno.invalid_reason IS NULL OR sno.invalid_reason = '')
        ORDER BY icd.concept_code
    """).fetchall()
    conn.close()

    pairs = [
        {"icd_code": r[0], "icd_label": r[1],
         "snomed_code": r[2], "snomed_label": r[3]}
        for r in rows
    ]
    if sample_n and sample_n < len(pairs):
        random.seed(42)
        pairs = random.sample(pairs, sample_n)
    return pairs


# ── OWL2Vec matcher ────────────────────────────────────────────────────────

def load_owl2vec():
    from gensim.models import Word2Vec
    wv = Word2Vec.load(OWL2VEC_MODEL).wv
    C  = "c:"
    concept_keys = [k for k in wv.key_to_index if k.startswith(C)]
    mat = np.vstack([wv[k] for k in concept_keys])
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    mat_normed = mat / norms
    return wv, concept_keys, mat_normed, C

def owl2vec_query(label: str, wv, concept_keys, mat_normed, C, topn=5):
    tokens = [t for t in re.sub(r"[^a-zA-Z0-9 ]"," ", label.lower()).split()
              if t not in STOPWORDS and len(t) > 2]
    vecs = [wv[t] for t in tokens if t in wv]
    if not vecs:
        return []
    q = np.mean(vecs, axis=0)
    norm = np.linalg.norm(q)
    if norm == 0:
        return []
    q /= norm
    scores = mat_normed @ q
    top_i  = np.argpartition(scores, -topn)[-topn:]
    top_i  = top_i[np.argsort(scores[top_i])[::-1]]
    return [(concept_keys[i][len(C):], float(scores[i])) for i in top_i]


# ── BioBERT matcher ────────────────────────────────────────────────────────

def load_biobert():
    from pathlib import Path
    if not Path(BIOBERT_IDS).exists():
        return None, None, None
    import torch
    from transformers import AutoTokenizer, AutoModel
    print("Loading BioBERT model…")
    tok   = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    model.eval()
    ids   = np.load(BIOBERT_IDS, allow_pickle=True).tolist()
    vecs  = np.load(BIOBERT_VECS).astype(np.float32)
    print(f"BioBERT index: {len(ids):,} concepts")
    return tok, model, ids, vecs

def biobert_query(label: str, tok, model, ids, vecs, topn=5):
    import torch
    inputs = tok([label], padding=True, truncation=True,
                 max_length=128, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    q = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    q = q.cpu().float().numpy()[0]
    q /= (np.linalg.norm(q) or 1)
    scores = vecs @ q
    top_i  = np.argpartition(scores, -topn)[-topn:]
    top_i  = top_i[np.argsort(scores[top_i])[::-1]]
    return [(ids[i], float(scores[i])) for i in top_i]


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(pairs: list[dict], query_fn, method_name: str, topn=5):
    hits = defaultdict(int)
    mrr_sum  = 0.0
    sim_when_found = []
    not_in_vocab   = 0
    total = len(pairs)

    t0 = time.time()
    for i, p in enumerate(pairs):
        if i % 500 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / (i + 1)) * (total - i) if i else 0
            print(f"  {i:>5}/{total}  ETA {eta/60:.1f} min", end="\r", flush=True)

        results = query_fn(p["icd_label"])  # list of (snomed_code, score)
        top_codes = [r[0] for r in results]
        target    = p["snomed_code"]

        if target not in top_codes:
            # Check if target is even in our index
            if not results:
                not_in_vocab += 1
            continue

        rank = top_codes.index(target) + 1          # 1-based
        for k in (1, 3, 5):
            if rank <= k:
                hits[k] += 1
        mrr_sum += 1.0 / rank
        sim_when_found.append(results[rank - 1][1])

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s                       ")

    found = hits[5]
    print(f"\n{'─'*52}")
    print(f"  Method : {method_name}")
    print(f"  Pairs  : {total:,}  (target not in vocab: {not_in_vocab})")
    print(f"  Hit@1  : {hits[1]/total:.1%}  ({hits[1]:,})")
    print(f"  Hit@3  : {hits[3]/total:.1%}  ({hits[3]:,})")
    print(f"  Hit@5  : {hits[5]/total:.1%}  ({hits[5]:,})")
    print(f"  MRR    : {mrr_sum/total:.4f}")
    if sim_when_found:
        print(f"  Avg sim (when found): {np.mean(sim_when_found):.4f}")
    print(f"{'─'*52}\n")

    return {"method": method_name, "total": total,
            "hit1": hits[1]/total, "hit3": hits[3]/total,
            "hit5": hits[5]/total, "mrr": mrr_sum/total}


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1000,
                    help="Number of ICD-10 pairs to evaluate (default 1000)")
    ap.add_argument("--method", choices=["owl2vec", "biobert", "both"],
                    default="both")
    args = ap.parse_args()

    print(f"Loading {args.sample:,} ICD-10→SNOMED test pairs…")
    pairs = load_test_pairs(args.sample)
    print(f"Loaded {len(pairs):,} pairs\n")

    results = []

    if args.method in ("owl2vec", "both"):
        print("Loading OWL2Vec model…")
        wv, concept_keys, mat_normed, C = load_owl2vec()
        print(f"OWL2Vec vocab: {len(concept_keys):,} concepts\n")
        print("Evaluating OWL2Vec…")
        fn = lambda lbl: owl2vec_query(lbl, wv, concept_keys, mat_normed, C, topn=5)
        results.append(evaluate(pairs, fn, "OWL2Vec mean-pool", topn=5))

    if args.method in ("biobert", "both"):
        from pathlib import Path
        if not Path(BIOBERT_IDS).exists():
            print("BioBERT index not built yet — skipping.")
        else:
            tok, model, ids, vecs = load_biobert()
            if tok:
                print("\nEvaluating BioBERT…")
                fn = lambda lbl: biobert_query(lbl, tok, model, ids, vecs, topn=5)
                results.append(evaluate(pairs, fn, "BioBERT (dmis-lab/biobert-base-cased-v1.2)", topn=5))

    # Side-by-side summary
    if len(results) > 1:
        print(f"\n{'Method':<45}  Hit@1    Hit@3    Hit@5    MRR")
        print("─" * 75)
        for r in results:
            print(f"  {r['method']:<43}  {r['hit1']:.1%}    {r['hit3']:.1%}    {r['hit5']:.1%}    {r['mrr']:.4f}")


if __name__ == "__main__":
    main()
