"""
OWL2Vec* implementation for the SNOMED OWL ontology.

Implements the three OWL2Vec* corpus types from the paper
(Chen et al., 2021 — https://arxiv.org/abs/2009.14654):
  1. Structural corpus  — random walks on the projected axiom graph
  2. Lexical corpus     — label tokens anchored to concept URIs
  3. Combined corpus    — interleaved structural + lexical walks

Runs entirely from the existing SQLite cache (snomed_cache.db),
so no re-parsing of the 185 MB OWL file is needed.

Usage:
    .venv311/bin/python owl2vec_train.py
or with options:
    .venv311/bin/python owl2vec_train.py --walks 50 --depth 4 --dim 200
"""

import argparse
import random
import re
import sqlite3
from pathlib import Path

from gensim.models import Word2Vec
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
DB_PATH    = "/Users/remzicelebi/workspace/snomed/snomed_cache.db"
MODEL_PATH = "/Users/remzicelebi/workspace/snomed/snomed_owl2vec.model"

# ── Token prefixes (keep URIs readable and distinct) ──────────────────────
C  = "c:"    # concept,  e.g. c:44054006
R  = "r:"    # role attr, e.g. r:363698007
RI = "ri:"   # inverse role
ISA    = "isa"
ISA_INV = "isa_inv"


def c(cid):  return C  + str(cid)
def r(aid):  return R  + str(aid)
def ri(aid): return RI + str(aid)


# ── Graph construction ─────────────────────────────────────────────────────

def build_graph(conn: sqlite3.Connection) -> dict:
    """
    Adjacency list: node → [(relation_token, neighbour_token), ...]
    Includes bidirectional hierarchy edges and role edges.
    """
    adj: dict[str, list] = {}

    # Hierarchy (SubClassOf simple + from intersection axioms)
    for child, parent in conn.execute(
        "SELECT child_id, parent_id FROM parents WHERE parent_id != '609096000'"
    ):
        ch, pa = c(child), c(parent)
        adj.setdefault(ch, []).append((ISA, pa))
        adj.setdefault(pa, []).append((ISA_INV, ch))

    # Role restrictions (ObjectSomeValuesFrom — the core OWL2Vec* signal)
    for concept, attr, val in conn.execute(
        "SELECT concept_id, attr_id, val_id FROM roles"
    ):
        co, rv, va = c(concept), r(attr), c(val)
        # Forward: concept --role_attr--> value
        adj.setdefault(co, []).append((rv, va))
        # Inverse: value --role_attr_inv--> concept
        adj.setdefault(va, []).append((ri(attr), co))

    return adj


# ── Walk generation ────────────────────────────────────────────────────────

def random_walk(adj: dict, start: str, depth: int) -> list[str]:
    """
    One structural random walk.
    Walk sentence: [start, rel1, node1, rel2, node2, ...]
    Relations as tokens encode the edge semantics in the Word2Vec window.
    """
    walk = [start]
    node = start
    for _ in range(depth):
        neighbours = adj.get(node, [])
        if not neighbours:
            break
        rel, nxt = random.choice(neighbours)
        walk.extend([rel, nxt])
        node = nxt
    return walk


def lexical_walk(concept_token: str, label: str) -> list[str]:
    """
    Lexical walk: concept URI followed by cleaned label tokens.
    Bridges structural and textual similarity.
    """
    tokens = re.sub(r"[^a-zA-Z0-9 ]", " ", label.lower()).split()
    if not tokens:
        return []
    return [concept_token] + tokens


def generate_corpus(
    conn: sqlite3.Connection,
    adj: dict,
    num_walks: int,
    depth: int,
) -> list[list[str]]:
    corpus: list[list[str]] = []

    rows = conn.execute(
        "SELECT id, COALESCE(pref_label, label) FROM concepts WHERE label IS NOT NULL"
    ).fetchall()

    print(
        f"[owl2vec] Generating walks: {len(rows):,} concepts × "
        f"{num_walks} walks × depth {depth}"
    )

    for cid, label in tqdm(rows, desc="walks"):
        node = c(cid)
        if node not in adj:
            continue

        # 1. Structural walks
        for _ in range(num_walks):
            w = random_walk(adj, node, depth)
            if len(w) > 1:
                corpus.append(w)

        # 2. Lexical walk (label → concept anchoring)
        lex = lexical_walk(node, label)
        if lex:
            corpus.append(lex)
            # 3. Combined: structural prefix + label tokens (OWL2Vec* corpus 3)
            structural_prefix = random_walk(adj, node, depth=2)
            corpus.append(structural_prefix + lex[1:])

    return corpus


# ── Training ───────────────────────────────────────────────────────────────

def train(
    db_path:    str = DB_PATH,
    model_path: str = MODEL_PATH,
    num_walks:  int = 50,
    depth:      int = 4,
    dim:        int = 200,
    window:     int = 5,
    epochs:     int = 10,
    workers:    int = 8,
) -> None:
    conn = sqlite3.connect(db_path)

    print("[owl2vec] Building axiom graph from SQLite cache…")
    adj = build_graph(conn)
    print(f"[owl2vec] Graph: {len(adj):,} nodes")

    corpus = generate_corpus(conn, adj, num_walks, depth)
    print(f"[owl2vec] Corpus: {len(corpus):,} walk sentences")

    print(f"[owl2vec] Training Word2Vec (dim={dim}, window={window}, epochs={epochs})…")
    model = Word2Vec(
        sentences=corpus,
        vector_size=dim,
        window=window,
        min_count=1,
        sg=1,           # skip-gram (better for rare tokens)
        workers=workers,
        epochs=epochs,
        seed=42,
    )
    model.save(model_path)
    vocab_size = len(model.wv)
    print(f"[owl2vec] Saved → {model_path}  (vocab: {vocab_size:,})")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train OWL2Vec* on SNOMED OWL cache")
    p.add_argument("--db",     default=DB_PATH,    help="SQLite cache path")
    p.add_argument("--out",    default=MODEL_PATH,  help="Output model path")
    p.add_argument("--walks",  type=int, default=50,  help="Walks per concept")
    p.add_argument("--depth",  type=int, default=4,   help="Walk depth")
    p.add_argument("--dim",    type=int, default=200,  help="Embedding dimension")
    p.add_argument("--window", type=int, default=5,   help="Word2Vec window")
    p.add_argument("--epochs", type=int, default=10,  help="Word2Vec epochs")
    p.add_argument("--workers",type=int, default=8,   help="Parallel workers")
    args = p.parse_args()
    train(args.db, args.out, args.walks, args.depth,
          args.dim, args.window, args.epochs, args.workers)
