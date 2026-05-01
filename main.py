import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from athena_client import AthenaLocal
from snomed_owl import SnomedOWL

VOCAB_DIR = "/Users/remzicelebi/Downloads/icd10tosnomed"
OWL_PATH = "/Users/remzicelebi/workspace/snomed/ontology-2024-01-05_15-30-19.owl"
OWL_DB   = "/Users/remzicelebi/workspace/snomed/snomed_cache.db"

athena = AthenaLocal(VOCAB_DIR)
snomed = SnomedOWL(OWL_PATH, OWL_DB)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, athena.ensure_cached)
    loop.run_in_executor(None, snomed.ensure_cached)
    yield


app = FastAPI(title="ICD-10 → SNOMED Mapper", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/api/status")
def status():
    return {
        "athena_ready":    athena.is_ready(),
        "athena_error":    athena.get_error(),
        "has_icd10cm":     athena.has_icd10cm() if athena.is_ready() else False,
        "owl_ready":       snomed.is_ready(),
        "owl_concepts":    snomed.concept_count() if snomed.is_ready() else 0,
    }


@app.get("/api/map/{icd10_code:path}")
async def map_code(icd10_code: str):
    code = icd10_code.strip().upper().replace(" ", "")

    if not athena.is_ready():
        return {"error": "Athena vocabulary cache is still loading — please wait a moment."}

    if not athena.has_icd10cm():
        return {
            "error": (
                "ICD10CM vocabulary not found in the downloaded files. "
                "Please re-download from athena.ohdsi.org with ICD10CM selected "
                "(see instructions in the app)."
            )
        }

    icd10 = athena.find_concept(code, "ICD10CM")
    if not icd10:
        return {"error": f"ICD-10 code '{code}' not found in the local vocabulary."}

    maps_to, maps_to_value = athena.get_snomed_mappings(icd10["id"])
    if maps_to:
        post_coords = []
        if snomed.is_ready():
            for s in maps_to:
                pc = snomed.suggest_post_coord(str(s.get("code") or ""))
                if pc:
                    post_coords.append(pc)
        return {
            "icd10":         icd10,
            "type":          "pre-coordinated",
            "snomed":        maps_to,
            "maps_to_value": maps_to_value,
            "post_coords":   post_coords,
            "owl_ready":     snomed.is_ready(),
        }

    # Walk up parent codes
    parent_code = code
    parent_icd10 = None
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
            "icd10":   icd10,
            "type":    "no-mapping",
            "message": "No SNOMED mapping found for this code or any of its parent codes.",
        }

    focus      = parent_snomed[0]
    snomed_code = str(focus.get("code") or "")
    post_coord = snomed.suggest_post_coord(snomed_code) if snomed.is_ready() else None

    return {
        "icd10":        icd10,
        "type":         "post-coordinated",
        "parent_icd10": parent_icd10,
        "focus_snomed": focus,
        "post_coord":   post_coord,
        "owl_ready":    snomed.is_ready(),
    }


def _parent(code: str) -> str:
    if "." in code:
        base, ext = code.split(".", 1)
        return (base + "." + ext[:-1]) if len(ext) > 1 else base
    return ""


@app.get("/", response_class=HTMLResponse)
def frontend():
    return Path("static/index.html").read_text()
