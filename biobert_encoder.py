"""
BioBERT-based SNOMED concept retrieval.

Encodes SNOMED concept labels with BioBERT (mean-pool of last hidden states),
builds a pre-computed L2-normalised index once, and at query time encodes the
ICD-10 label to find the nearest SNOMED concepts by cosine similarity.
"""

import sqlite3
import threading
from pathlib import Path
from typing import Optional

import numpy as np

SNOMED_DB  = "/Users/remzicelebi/workspace/snomed/snomed_cache.db"
ATHENA_DB  = "/Users/remzicelebi/workspace/snomed/athena_vocab_cache.db"
INDEX_IDS  = "/Users/remzicelebi/workspace/snomed/biobert_ids.npy"
INDEX_VECS = "/Users/remzicelebi/workspace/snomed/biobert_vecs.npy"
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"


class BioBERTEncoder:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        ids_path:   str = INDEX_IDS,
        vecs_path:  str = INDEX_VECS,
        db_path:    str = SNOMED_DB,
    ):
        self.model_name = model_name
        self.ids_path   = ids_path
        self.vecs_path  = vecs_path
        self.db_path    = db_path

        self._tokenizer = None
        self._model     = None
        self._device    = None
        self._ids:  Optional[list]       = None
        self._vecs: Optional[np.ndarray] = None

        self._ready = threading.Event()
        self._lock  = threading.Lock()
        self._error: Optional[str] = None

    def is_ready(self)  -> bool:          return self._ready.is_set()
    def get_error(self) -> Optional[str]: return self._error

    # ── Startup ────────────────────────────────────────────────────────

    def ensure_loaded(self):
        with self._lock:
            if self._ready.is_set():
                return
            if not (Path(self.ids_path).exists() and Path(self.vecs_path).exists()):
                self._error = (
                    "BioBERT index not found. "
                    "Run: .venv311/bin/python build_biobert_index.py"
                )
                return
            try:
                self._load_model()
                self._load_index()
                self._ready.set()
            except Exception as e:
                self._error = str(e)

    def _load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModel

        print(f"[biobert] Loading {self.model_name}…")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model     = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._device = torch.device("cpu")
        self._model  = self._model.to(self._device)
        print("[biobert] Model ready.")

    def _load_index(self):
        size_mb = Path(self.vecs_path).stat().st_size / 1e6
        print(f"[biobert] Loading index ({size_mb:.0f} MB)…")
        self._ids  = np.load(self.ids_path, allow_pickle=True).tolist()
        self._vecs = np.load(self.vecs_path)   # (N, 768) float16, L2-normalised
        self._vecs = self._vecs.astype(np.float32)
        print(f"[biobert] Index ready — {len(self._ids):,} concepts, dim={self._vecs.shape[1]}")

    # ── Encoding ──────────────────────────────────────────────────────

    def _encode(self, texts: list[str]) -> np.ndarray:
        import torch
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vecs = vecs.cpu().float().numpy()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms

    # ── Query ──────────────────────────────────────────────────────────

    def match_by_label(self, icd10_label: str, topn: int = 5) -> list[dict]:
        """
        Encode ICD-10 label with BioBERT and return the topn most similar
        SNOMED concepts by cosine similarity against the pre-built index.
        """
        if not self._ready.is_set():
            return []
        q      = self._encode([icd10_label])[0]    # (768,) normalised
        scores = self._vecs @ q                    # (N,)
        top_i  = np.argpartition(scores, -topn)[-topn:]
        top_i  = top_i[np.argsort(scores[top_i])[::-1]]
        return [
            {
                "id":    self._ids[i],
                "label": self._snomed_label(self._ids[i]),
                "score": round(float(scores[i]), 4),
            }
            for i in top_i
        ]

    def _snomed_label(self, cid: str) -> str:
        with sqlite3.connect(self.db_path, check_same_thread=False) as c:
            row = c.execute(
                "SELECT COALESCE(pref_label, label, ?) FROM concepts WHERE id=?",
                (cid, cid),
            ).fetchone()
        return row[0] if row else cid
