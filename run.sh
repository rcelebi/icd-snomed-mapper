#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Python 3.11 venv — needed for gensim (OWL2Vec)
if [ ! -d ".venv311" ]; then
  echo "Creating Python 3.11 venv..."
  python3.11 -m venv .venv311
  .venv311/bin/pip install -q \
    fastapi "uvicorn[standard]" httpx \
    "gensim==4.3.2" "numpy<2" "scipy<1.13" tqdm
fi

MODEL="/Users/remzicelebi/workspace/snomed/snomed_owl2vec.model"
if [ ! -f "$MODEL" ]; then
  echo ""
  echo "OWL2Vec model not found. Train it first (runs in background, ~35 min):"
  echo "  .venv311/bin/python owl2vec_train.py"
  echo ""
fi

echo "Starting ICD-10 → SNOMED Mapper on http://localhost:8000"
.venv311/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
