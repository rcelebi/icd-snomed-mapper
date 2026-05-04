import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from athena_client import AthenaLocal
from snomed_owl import SnomedOWL
from owl2vec_recommender import OWL2VecRecommender
from biobert_encoder import BioBERTEncoder

VOCAB_DIR = "/Users/remzicelebi/Downloads/icd10tosnomed"
OWL_PATH  = "/Users/remzicelebi/workspace/snomed/ontology-2024-01-05_15-30-19.owl"
OWL_DB    = "/Users/remzicelebi/workspace/snomed/snomed_cache.db"

athena  = AthenaLocal(VOCAB_DIR)
snomed  = SnomedOWL(OWL_PATH, OWL_DB)
owl2vec = OWL2VecRecommender()
biobert = BioBERTEncoder()


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, athena.ensure_cached)
    loop.run_in_executor(None, snomed.ensure_cached)
    loop.run_in_executor(None, owl2vec.ensure_loaded)
    loop.run_in_executor(None, biobert.ensure_loaded)
    yield


app = FastAPI(title="ICD-10 → SNOMED Mapper", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/api/status")
def status():
    return {
        "athena_ready":  athena.is_ready(),
        "athena_error":  athena.get_error(),
        "has_icd10cm":   athena.has_icd10cm() if athena.is_ready() else False,
        "owl_ready":     snomed.is_ready(),
        "owl_concepts":  snomed.concept_count() if snomed.is_ready() else 0,
        "owl2vec_ready": owl2vec.is_ready(),
        "owl2vec_error": owl2vec.get_error(),
        "biobert_ready": biobert.is_ready(),
        "biobert_error": biobert.get_error(),
    }


@app.get("/api/map/{icd10_code:path}")
async def map_code(icd10_code: str):
    code = icd10_code.strip().upper().replace(" ", "")

    if not athena.is_ready():
        return {"error": "Athena vocabulary cache is still loading — please wait."}
    if not athena.has_icd10cm():
        return {"error": "ICD10CM not in local vocabulary. Re-download from athena.ohdsi.org."}

    icd10 = athena.find_concept(code, "ICD10CM")
    if not icd10:
        return {"error": f"ICD-10 code '{code}' not found in the local vocabulary."}

    icd10_label = icd10.get("name") or code

    # ── Two-stage vector match (always computed) ──────────────────────────
    # Stage 1: BioBERT finds top-K similar SNOMED pre-coordinated concepts
    # Stage 2: OWL2Vec aggregates post-coordinated patterns from those K concepts
    vector_match = None
    if biobert.is_ready() and owl2vec.is_ready() and snomed.is_ready():
        biobert_hits = biobert.match_by_label(icd10_label, topn=5)
        if biobert_hits:
            post_coord_suggestion = owl2vec.suggest_post_coord_from_biobert(
                biobert_hits, snomed, topn_values=5
            )
            vector_match = {
                "icd10_label": icd10_label,
                "method":      "biobert+owl2vec",
                "biobert_hits": biobert_hits,
                "post_coord":   post_coord_suggestion,
            }
    elif owl2vec.is_ready() and snomed.is_ready():
        # Fallback to OWL2Vec-only label match when BioBERT index not built
        vector_match = owl2vec.suggest_post_coord_from_label(
            icd10_label, snomed, topn_concepts=5, topn_values=3
        )

    # ── Athena direct mapping ─────────────────────────────────────────────
    maps_to, maps_to_value = athena.get_snomed_mappings(icd10["id"])
    if maps_to:
        post_coords = []
        if snomed.is_ready():
            for s in maps_to:
                sc = str(s.get("code") or "")
                pc = snomed.suggest_post_coord(sc)
                if pc:
                    pc = owl2vec.enrich_post_coord(sc, pc)
                    post_coords.append(pc)
        return {
            "icd10":          icd10,
            "type":           "pre-coordinated",
            "snomed":         maps_to,
            "maps_to_value":  maps_to_value,
            "post_coords":    post_coords,
            "vector_match":   vector_match,
            "owl_ready":      snomed.is_ready(),
            "owl2vec_ready":  owl2vec.is_ready(),
            "biobert_ready":  biobert.is_ready(),
        }

    # ── Walk up parent codes ──────────────────────────────────────────────
    parent_code   = code
    parent_icd10  = None
    parent_snomed: list = []
    while True:
        parent_code = _parent(parent_code)
        if not parent_code:
            break
        parent_icd10 = athena.find_concept(parent_code, "ICD10CM")
        if parent_icd10:
            parent_snomed, _ = athena.get_snomed_mappings(parent_icd10["id"])
            if parent_snomed:
                break

    if not parent_snomed:
        return {
            "icd10":         icd10,
            "type":          "no-mapping",
            "vector_match":  vector_match,
            "owl2vec_ready": owl2vec.is_ready(),
            "biobert_ready": biobert.is_ready(),
        }

    focus       = parent_snomed[0]
    snomed_code = str(focus.get("code") or "")
    post_coord  = None
    if snomed.is_ready():
        post_coord = snomed.suggest_post_coord(snomed_code)
        if post_coord:
            post_coord = owl2vec.enrich_post_coord(snomed_code, post_coord)

    return {
        "icd10":          icd10,
        "type":           "post-coordinated",
        "parent_icd10":   parent_icd10,
        "focus_snomed":   focus,
        "post_coord":     post_coord,
        "vector_match":   vector_match,
        "owl_ready":      snomed.is_ready(),
        "owl2vec_ready":  owl2vec.is_ready(),
        "biobert_ready":  biobert.is_ready(),
    }


def _parent(code: str) -> str:
    if "." in code:
        base, ext = code.split(".", 1)
        return (base + "." + ext[:-1]) if len(ext) > 1 else base
    return ""


@app.get("/", response_class=HTMLResponse)
def frontend():
    return Path("static/index.html").read_text()
