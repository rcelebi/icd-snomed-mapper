"""
Pre-compute BioBERT embeddings for all standard SNOMED concepts.
Run once — takes ~16 min on CPU (8-core).

    .venv311/bin/python build_biobert_index.py
"""

import sqlite3
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

ATHENA_DB  = "/Users/remzicelebi/workspace/snomed/athena_vocab_cache.db"
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
BATCH_SIZE = 128
MAX_LEN    = 64
IDS_OUT    = "/Users/remzicelebi/workspace/snomed/biobert_ids.npy"
VECS_OUT   = "/Users/remzicelebi/workspace/snomed/biobert_vecs.npy"


def main():
    # Load standard SNOMED concepts from Athena vocab
    conn = sqlite3.connect(ATHENA_DB)
    rows = conn.execute(
        """SELECT concept_code, concept_name
           FROM concepts
           WHERE vocabulary_id = 'SNOMED'
             AND standard = 'S'
             AND (invalid_reason IS NULL OR invalid_reason = '')
           ORDER BY concept_code"""
    ).fetchall()
    conn.close()

    ids    = [r[0] for r in rows]
    labels = [r[1] for r in rows]
    print(f"Concepts to encode: {len(ids):,}")

    # Load model
    print(f"Loading {MODEL_NAME}…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    device = torch.device("cpu")
    model  = model.to(device)
    print(f"Device: {device}, threads: {torch.get_num_threads()}")

    # Encode in batches
    all_vecs = []
    for i in tqdm(range(0, len(labels), BATCH_SIZE), desc="encoding"):
        batch  = labels[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(**inputs)

        # Mean pooling over non-padding tokens
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        all_vecs.append(vecs.cpu().float().numpy())

    vecs = np.vstack(all_vecs).astype(np.float32)

    # L2-normalise
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vecs /= norms

    # Save as float16 to halve disk usage
    np.save(IDS_OUT,  np.array(ids,  dtype=object))
    np.save(VECS_OUT, vecs.astype(np.float16))

    size_mb = vecs.astype(np.float16).nbytes / 1e6
    print(f"\nSaved {vecs.shape[0]:,} vectors ({size_mb:.0f} MB)")
    print(f"  IDs  → {IDS_OUT}")
    print(f"  Vecs → {VECS_OUT}")


if __name__ == "__main__":
    main()
