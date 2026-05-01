"""
Loads Athena vocabulary CSVs (CONCEPT.csv + CONCEPT_RELATIONSHIP.csv)
into a local SQLite cache and provides fast ICD-10 → SNOMED lookups.
"""

import csv
import sqlite3
import threading
from pathlib import Path
from typing import Optional

DB_PATH = "/Users/remzicelebi/workspace/snomed/athena_vocab_cache.db"


class AthenaLocal:
    def __init__(self, vocab_dir: str):
        self.vocab_dir = Path(vocab_dir)
        self.db_path   = DB_PATH
        self._ready    = threading.Event()
        self._lock     = threading.Lock()
        self._error: Optional[str] = None

    def is_ready(self) -> bool:
        return self._ready.is_set()

    def get_error(self) -> Optional[str]:
        return self._error

    def ensure_cached(self):
        with self._lock:
            if Path(self.db_path).exists():
                self._ready.set()
                return
            try:
                self._build_cache()
                self._ready.set()
            except Exception as e:
                self._error = str(e)

    def _conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _build_cache(self):
        concept_csv  = self.vocab_dir / "CONCEPT.csv"
        relation_csv = self.vocab_dir / "CONCEPT_RELATIONSHIP.csv"
        if not concept_csv.exists():
            raise FileNotFoundError(f"CONCEPT.csv not found in {self.vocab_dir}")
        if not relation_csv.exists():
            raise FileNotFoundError(f"CONCEPT_RELATIONSHIP.csv not found in {self.vocab_dir}")

        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS concepts (
                concept_id      INTEGER PRIMARY KEY,
                concept_name    TEXT,
                domain_id       TEXT,
                vocabulary_id   TEXT,
                concept_class   TEXT,
                standard        TEXT,
                concept_code    TEXT,
                invalid_reason  TEXT
            );
            CREATE TABLE IF NOT EXISTS relationships (
                concept_id_1    INTEGER,
                concept_id_2    INTEGER,
                relationship_id TEXT,
                invalid_reason  TEXT
            );
        """)

        print("[athena] Loading CONCEPT.csv …")
        buf = []
        with open(concept_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                buf.append((
                    int(row["concept_id"]),
                    row["concept_name"],
                    row["domain_id"],
                    row["vocabulary_id"],
                    row["concept_class_id"],
                    row["standard_concept"],
                    row["concept_code"],
                    row["invalid_reason"],
                ))
                if len(buf) >= 50_000:
                    conn.executemany(
                        "INSERT OR REPLACE INTO concepts VALUES(?,?,?,?,?,?,?,?)", buf
                    )
                    conn.commit()
                    buf.clear()
        if buf:
            conn.executemany(
                "INSERT OR REPLACE INTO concepts VALUES(?,?,?,?,?,?,?,?)", buf
            )
            conn.commit()

        print("[athena] Loading CONCEPT_RELATIONSHIP.csv …")
        buf = []
        with open(relation_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("invalid_reason"):
                    continue
                rel = row["relationship_id"]
                if rel not in ("Maps to", "Maps to value"):
                    continue
                buf.append((
                    int(row["concept_id_1"]),
                    int(row["concept_id_2"]),
                    rel,
                    row.get("invalid_reason", ""),
                ))
                if len(buf) >= 50_000:
                    conn.executemany(
                        "INSERT INTO relationships VALUES(?,?,?,?)", buf
                    )
                    conn.commit()
                    buf.clear()
        if buf:
            conn.executemany("INSERT INTO relationships VALUES(?,?,?,?)", buf)
            conn.commit()

        conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_concept_vocab_code
                ON concepts(vocabulary_id, concept_code);
            CREATE INDEX IF NOT EXISTS idx_rel_src
                ON relationships(concept_id_1, relationship_id);
        """)
        conn.commit()
        conn.close()
        print("[athena] Vocab cache ready.")

    def find_concept(self, code: str, vocab: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute(
                """SELECT concept_id, concept_name, domain_id, vocabulary_id,
                          concept_class, standard, concept_code
                   FROM concepts
                   WHERE vocabulary_id=? AND UPPER(concept_code)=?
                     AND (invalid_reason IS NULL OR invalid_reason='')
                   LIMIT 1""",
                (vocab, code.upper()),
            ).fetchone()
            if not row:
                return None
            return {
                "id":            row[0],
                "name":          row[1],
                "domainId":      row[2],
                "vocabularyId":  row[3],
                "conceptClassId": row[4],
                "standardConcept": row[5],
                "code":          row[6],
            }

    def get_snomed_mappings(self, concept_id: int) -> tuple[list, list]:
        maps_to, maps_to_value = [], []
        with self._conn() as c:
            rows = c.execute(
                """SELECT r.relationship_id,
                          c.concept_id, c.concept_name, c.domain_id,
                          c.vocabulary_id, c.concept_class, c.standard, c.concept_code
                   FROM relationships r
                   JOIN concepts c ON c.concept_id = r.concept_id_2
                   WHERE r.concept_id_1=?
                     AND c.vocabulary_id='SNOMED'
                     AND (r.invalid_reason IS NULL OR r.invalid_reason='')""",
                (concept_id,),
            ).fetchall()
            for row in rows:
                rel, cid, name, dom, vocab, cls, std, code = row
                concept = {
                    "id": cid, "name": name, "domainId": dom,
                    "vocabularyId": vocab, "conceptClassId": cls,
                    "standardConcept": std, "code": code,
                }
                if rel == "Maps to value":
                    maps_to_value.append(concept)
                else:
                    maps_to.append(concept)
        return maps_to, maps_to_value

    def has_icd10cm(self) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM concepts WHERE vocabulary_id='ICD10CM' LIMIT 1"
            ).fetchone()
            return row is not None
