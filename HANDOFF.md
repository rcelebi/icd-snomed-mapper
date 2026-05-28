# ICD-10 → SNOMED Mapper — Handoff

**Repo**: https://github.com/rcelebi/icd-snomed-mapper  
**Stack**: FastAPI · Python 3.11 · SQLite · BioBERT · OWL2Vec* · Gensim · Transformers

---

## What it does

Two-panel web app that maps ICD-10-CM codes to SNOMED CT with post-coordinated expressions.

- **Left panel (Direct)**: Athena OMOP `Maps to` relationship → SNOMED concept + OWL-derived post-coordinated expression enriched with OWL2Vec analogy rankings
- **Right panel (Vector)**: BioBERT label similarity → top-5 SNOMED concepts → OWL2Vec consensus post-coordinated expression

---

## Pipeline

```
ICD-10 code
  │
  ├─ athena_client.py ──── CONCEPT.csv + CONCEPT_RELATIONSHIP.csv
  │   find_concept()           SQLite cache: athena_vocab_cache.db
  │   get_snomed_mappings()    "Maps to" / "Maps to value" relationships
  │
  ├─ snomed_owl.py ────── OWL Functional Syntax (185 MB, SNOMED 2024-01-05)
  │   ensure_cached()         SQLite cache: snomed_cache.db
  │   suggest_post_coord()    concept's own OWL roles + child refinements
  │
  ├─ [Stage 1] biobert_encoder.py
  │   match_by_label()        encode ICD-10 label → 768-dim → cosine search
  │                           pre-built index: biobert_ids.npy + biobert_vecs.npy
  │                           349K SNOMED labels, float16, ~512 MB
  │
  └─ [Stage 2] owl2vec_recommender.py
      suggest_post_coord_from_biobert()
        1. for each BioBERT top-5 concept: get OWL own-roles + OWL2Vec vector
        2. mean-pool vectors → consensus embedding
        3. for each attribute: consensus + role_attr_vec ≈ best_value_vec
        4. agreement score = how many of 5 concepts share that attribute
```

---

## Local data files (not in repo)

All paths are hardcoded throughout the Python files.

| File | Path | How it was built |
|---|---|---|
| Athena vocab CSVs | `~/Downloads/icd10tosnomed/CONCEPT.csv` + `CONCEPT_RELATIONSHIP.csv` | Downloaded from athena.ohdsi.org (ICD10CM + SNOMED bundle) |
| Athena SQLite cache | `~/workspace/snomed/athena_vocab_cache.db` | Built automatically on first app start from the CSVs (~1-2 min) |
| SNOMED OWL | `~/workspace/snomed/ontology-2024-01-05_15-30-19.owl` | Downloaded from SNOMED International, 2024-01-05 release, 185 MB |
| SNOMED SQLite cache | `~/workspace/snomed/snomed_cache.db` | Built automatically on first app start from the OWL file (~60s) |
| OWL2Vec model | `~/workspace/snomed/snomed_owl2vec.model` | `python owl2vec_train.py` — 364K concepts, 50 walks, dim=200, ~20 min |
| BioBERT index IDs | `~/workspace/snomed/biobert_ids.npy` | `python build_biobert_index.py` — 349K standard SNOMED concept IDs |
| BioBERT index vecs | `~/workspace/snomed/biobert_vecs.npy` | Same script — 349K × 768 float16, L2-normalised, ~16 min |

**Why local files**: The Athena REST API returns 403 on all non-browser (server-side) requests.

The SQLite caches are rebuilt automatically if missing. The OWL2Vec model and BioBERT index must be built manually before those features are available.

---

## File breakdown

### `main.py` (166 lines)
FastAPI entry point. On startup, loads all four components in thread-pool executors (non-blocking). The `/api/map/{icd10_code}` endpoint always computes the two-stage vector match regardless of Athena result type, so both panels are always populated if models are ready. Three response types: `pre-coordinated`, `post-coordinated` (parent walk), `no-mapping`.

### `athena_client.py` (191 lines)
`AthenaLocal` class. Reads `CONCEPT.csv` + `CONCEPT_RELATIONSHIP.csv` into SQLite in 50K-row batches. Only stores `Maps to` and `Maps to value` relationships to keep the DB small. Key methods: `find_concept(code, vocab)`, `get_snomed_mappings(concept_id)`.

### `snomed_owl.py` (267 lines)
Parses SNOMED OWL Functional Syntax line-by-line with regex into three SQLite tables: `concepts` (id, label, pref_label), `parents` (child_id, parent_id), `roles` (concept_id, attr_id, val_id).

**Critical implementation note**: `_strip_some_values()` strips `ObjectSomeValuesFrom(...)` blocks before extracting bare parent IDs. The string `"ObjectSomeValuesFrom("` is **21 chars** — the comparison must be `s[i:i+21]`. An off-by-one (`i+20`) silently skips all stripping and pollutes the `parents` table with role and value IDs as fake parents, breaking all hierarchy lookups. This was found and fixed; if you rebuild the cache you must delete `snomed_cache.db` first.

`suggest_post_coord()` returns:
- `own_roles`: concept's own defining roles from the `roles` table
- `parents`: true ISA parents (filtered to exclude leaked role/value IDs)
- `own_expression`: SNOMED compositional grammar string
- `attributes`: refinements available from child concepts
- `example_children`: up to 6 child concepts with their expressions

### `owl2vec_train.py` (196 lines)
OWL2Vec* (Chen et al. 2021) implemented from scratch using the SQLite cache (the `owl2vec-star` PyPI package requires Python <3.9, incompatible with Apple Silicon + Python 3.11).

Three corpus types interleaved per concept:
1. **Structural**: random walk on axiom graph — `c:ID isa c:ID r:attrID c:ID …`
2. **Lexical**: `c:ID token1 token2 …` (bridges text and structure)
3. **Combined**: structural prefix + label tokens

Token prefixes: `c:` = concept, `r:` = role attribute, `ri:` = inverse role.  
Params: 50 walks/concept, depth 4, dim 200, window 5, 10 epochs, skip-gram.  
Output vocab: 436K tokens.

### `owl2vec_recommender.py` (380 lines)
`OWL2VecRecommender` class. Core method `rank_values(focus, attr, topn)` implements the analogy: `focus_vec + role_attr_vec ≈ value_vec`, scored by cosine similarity over all ontologically valid values for that attribute.

`suggest_post_coord_from_biobert()` is Stage 2 of the pipeline:
- Collects OWL own-roles and OWL2Vec vectors for each BioBERT top-K concept
- Mean-pools vectors → consensus embedding
- Runs the analogy for every attribute that appears in any concept's roles
- `agreement` = count of top-K concepts that define that attribute
- `in_owl: True` marks when the OWL2Vec-predicted value actually appears in the OWL roles of the retrieved concepts

### `biobert_encoder.py` (137 lines)
`BioBERTEncoder` class. Uses `dmis-lab/biobert-base-cased-v1.2`. Encoding: tokenise → BioBERT last hidden states → mean-pool over non-padding tokens → L2-normalise. Query time: `vecs @ query` (matrix-vector dot product on pre-normalised float32 index).

### `build_biobert_index.py` (86 lines)
One-time script. Queries Athena SQLite for all standard SNOMED concepts (`standard_concept = 'S'`), encodes labels in batches of 128 (max_length=64), saves float16 arrays.

### `evaluate_matching.py` (231 lines)
Hit@K / MRR evaluation on ICD-10→SNOMED ground-truth pairs from Athena. Run against 2,000 pairs:

| Method | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| OWL2Vec mean-pool | 0.8% | 1.7% | 2.4% | 0.013 |
| BioBERT | 12.7% | 17.0% | 19.6% | 0.152 |

BioBERT is ~8× better. The 80% miss rate at Hit@5 is mostly granularity mismatch: BioBERT retrieves more specific SNOMED concepts than the standard Athena mapping targets.

### `static/index.html`
Single-page Vue 3 (CDN) app. Two-panel layout. Left panel renders Athena direct mapping with OWL expression and role pills (checkmark badges when OWL value matches OWL2Vec top-1). Right panel renders BioBERT top-5 hits with similarity scores, then OWL2Vec consensus expression with agreement scores and `[OWL✓]` markers.

Frontend branching: right panel detects `vm.method === 'biobert+owl2vec'` (not a `biobert_ready` flag) to decide which template to render.

---

## Running

### Native
```bash
# Python 3.11 required (gensim)
.venv311/bin/uvicorn main:app --host 0.0.0.0 --port 8000

# Or with auto-reload for dev
bash run.sh
```

### Docker
```bash
docker compose up        # builds + starts, mounts data dirs as volumes
docker compose up -d     # detached
docker compose logs -f   # follow logs (watch for biobert/owl2vec ready messages)
```

Data volumes mounted read-only at the same absolute paths the code expects:
- `/Users/remzicelebi/workspace/snomed` → SNOMED cache, OWL2Vec model, BioBERT index
- `/Users/remzicelebi/Downloads/icd10tosnomed` → Athena CSVs

HuggingFace model downloads are cached in a named Docker volume (`huggingface_cache`).

Startup sequence (all async):
1. Athena + SNOMED SQLite caches load first (~few seconds if already built)
2. OWL2Vec model loads (~10s)
3. BioBERT model loads + index reads into RAM (~30-60s, ~512 MB)

Check `/api/status` to see readiness of each component.

---

## Known issues / bugs fixed

1. **`_strip_some_values` off-by-one** (`snomed_owl.py:21`) — `"ObjectSomeValuesFrom("` is 21 chars not 20. Fixed. If the snomed_cache.db was built with the broken version, delete it and let it rebuild.

2. **`biobert_ready` missing from API responses** — was absent from `pre-coordinated` and `post-coordinated` response types, causing the frontend to fall into the wrong rendering branch. Fixed in `main.py` — all three response types now include `biobert_ready`.

3. **Athena API 403** — All server-side requests to athena.ohdsi.org are blocked. The entire pipeline uses local Athena vocabulary CSV downloads instead.

---

## Potential next steps

- **Improve Hit@5**: Fine-tune BioBERT on ICD-10→SNOMED pairs (the Athena `Maps to` pairs are ground-truth training data). Expected large gain.
- **OMOP CDM integration**: Export mappings as CONCEPT_RELATIONSHIP rows for use in an OMOP database.
- **Make paths configurable**: Replace hardcoded absolute paths with env vars or a config file so the app is portable.
- **Batch API**: Add a `/api/batch` endpoint accepting multiple ICD-10 codes.
- **SNOMED release update**: The OWL file and Athena download are from 2024-01-05. Rebuilding the caches with a newer release requires deleting both `.db` files and re-running `build_biobert_index.py` and `owl2vec_train.py`.
