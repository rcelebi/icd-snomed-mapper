import re
import sqlite3
import threading
from pathlib import Path
from typing import Optional

RDFS_LABEL = re.compile(r'AnnotationAssertion\(rdfs:label :(\d+) "([^"]*)"@en\)')
PREF_LABEL = re.compile(r'AnnotationAssertion\(skos:prefLabel :(\d+) "([^"]*)"@en\)')
SIMPLE_SC  = re.compile(r'^SubClassOf\(:(\d+) :(\d+)\)\s*$')
SOME_VAL   = re.compile(r'ObjectSomeValuesFrom\(:(\d+) :(\d+)\)')
ROLE_GROUP = "609096000"
AXIOM_RE   = re.compile(r'^(?:SubClassOf|EquivalentClasses)\(:(\d+) ')


_OSV = "ObjectSomeValuesFrom("   # 21 chars

def _strip_some_values(s: str) -> str:
    """Remove all ObjectSomeValuesFrom(...) blocks, handling nested parens."""
    out, i, n = [], 0, len(s)
    while i < n:
        if s[i : i + 21] == _OSV:
            depth, i = 1, i + 21
            while i < n and depth:
                if   s[i] == "(": depth += 1
                elif s[i] == ")": depth -= 1
                i += 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


class SnomedOWL:
    def __init__(self, owl_path: str, db_path: str):
        self.owl_path = owl_path
        self.db_path  = db_path
        self._ready   = threading.Event()
        self._lock    = threading.Lock()

    def is_ready(self) -> bool:
        return self._ready.is_set()

    def concept_count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]

    def ensure_cached(self):
        with self._lock:
            if not Path(self.db_path).exists():
                print("[snomed] Building SQLite cache from OWL — first run takes ~60s …")
                self._build_cache()
                print("[snomed] Cache ready.")
            self._ready.set()

    def _conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _build_cache(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY, label TEXT, pref_label TEXT
            );
            CREATE TABLE IF NOT EXISTS parents (child_id TEXT, parent_id TEXT);
            CREATE TABLE IF NOT EXISTS roles   (concept_id TEXT, attr_id TEXT, val_id TEXT);
        """)

        labels, pref_labels = {}, {}
        par_buf, role_buf   = [], []
        BATCH = 40_000
        count = 0

        def flush():
            nonlocal count
            if labels:
                conn.executemany(
                    "INSERT OR REPLACE INTO concepts(id, label) VALUES(?,?)",
                    list(labels.items()),
                )
                labels.clear()
            if pref_labels:
                conn.executemany(
                    "INSERT OR IGNORE INTO concepts(id) VALUES(?)",
                    [(k,) for k in pref_labels],
                )
                conn.executemany(
                    "UPDATE concepts SET pref_label=? WHERE id=?",
                    [(v, k) for k, v in pref_labels.items()],
                )
                pref_labels.clear()
            if par_buf:
                conn.executemany("INSERT OR IGNORE INTO parents VALUES(?,?)", par_buf)
                par_buf.clear()
            if role_buf:
                conn.executemany("INSERT INTO roles VALUES(?,?,?)", role_buf)
                role_buf.clear()
            conn.commit()
            count = 0

        with open(self.owl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                m = RDFS_LABEL.match(line)
                if m:
                    labels[m.group(1)] = m.group(2)
                    count += 1
                    if count >= BATCH:
                        flush()
                    continue

                m = PREF_LABEL.match(line)
                if m:
                    pref_labels[m.group(1)] = m.group(2)
                    continue

                if not line.startswith(("SubClassOf(", "EquivalentClasses(")):
                    continue

                m = SIMPLE_SC.match(line)
                if m:
                    par_buf.append((m.group(1), m.group(2)))
                    continue

                m = AXIOM_RE.match(line)
                if not m:
                    continue
                cid = m.group(1)

                # Role restrictions: ObjectSomeValuesFrom(:attr :val)
                for attr, val in SOME_VAL.findall(line):
                    if attr != ROLE_GROUP:
                        role_buf.append((cid, attr, val))

                # Direct parents: bare :ID refs remaining after removing role expressions
                cleaned = _strip_some_values(line)
                for pid in re.findall(r":(\d+)", cleaned):
                    if pid != cid:
                        par_buf.append((cid, pid))

        flush()
        conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_par_child  ON parents(child_id);
            CREATE INDEX IF NOT EXISTS idx_par_parent ON parents(parent_id);
            CREATE INDEX IF NOT EXISTS idx_role_cid   ON roles(concept_id);
            CREATE INDEX IF NOT EXISTS idx_role_attr  ON roles(attr_id);
        """)
        conn.commit()
        conn.close()

    def get_label(self, cid: str) -> str:
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(pref_label, label, ?) FROM concepts WHERE id=?",
                (cid, cid),
            ).fetchone()
            return row[0] if row else cid

    def suggest_post_coord(self, snomed_code: str) -> Optional[dict]:
        with self._conn() as c:
            focus_label = self.get_label(snomed_code)

            # ── 1. Concept's own defining roles ──────────────────────────────
            own_role_rows = c.execute(
                "SELECT attr_id, val_id FROM roles WHERE concept_id=?",
                (snomed_code,),
            ).fetchall()
            own_roles = [
                {
                    "attr": {"id": a, "label": self.get_label(a)},
                    "val":  {"id": v, "label": self.get_label(v)},
                }
                for a, v in own_role_rows
            ]

            # ── 2. True parents (filter out role/value IDs leaked by old bug) ─
            role_ids = {r[0] for r in own_role_rows} | {r[1] for r in own_role_rows}
            role_ids.add("609096000")  # Role group
            parent_rows = c.execute(
                "SELECT DISTINCT parent_id FROM parents WHERE child_id=?",
                (snomed_code,),
            ).fetchall()
            parents = [
                {"id": pid, "label": self.get_label(pid)}
                for (pid,) in parent_rows
                if pid not in role_ids and pid != snomed_code
            ]

            # ── 3. Own post-coordinated expression ───────────────────────────
            own_expression = None
            if own_roles and parents:
                fp = parents[0]
                parts = [
                    f"{r['attr']['id']} | {r['attr']['label']} | = "
                    f"{r['val']['id']} | {r['val']['label']} |"
                    for r in own_roles
                ]
                own_expression = (
                    f"{fp['id']} | {fp['label']} | :\n"
                    f"  {{ {chr(10) + '    , '.join(parts)} }}"
                )

            # ── 4. Refinements available from children ────────────────────────
            child_rows = c.execute(
                "SELECT DISTINCT child_id FROM parents WHERE parent_id=? LIMIT 300",
                (snomed_code,),
            ).fetchall()
            child_ids = [r[0] for r in child_rows]

            own_attr_ids = {r[0] for r in own_role_rows}
            attributes = []
            if child_ids:
                ph = ",".join("?" * len(child_ids))
                attr_rows = c.execute(
                    f"SELECT DISTINCT attr_id FROM roles WHERE concept_id IN ({ph})",
                    child_ids,
                ).fetchall()
                for (attr_id,) in attr_rows:
                    vals = c.execute(
                        f"SELECT DISTINCT val_id FROM roles "
                        f"WHERE concept_id IN ({ph}) AND attr_id=? LIMIT 15",
                        [*child_ids, attr_id],
                    ).fetchall()
                    attributes.append({
                        "attr":    {"id": attr_id, "label": self.get_label(attr_id)},
                        "values":  [{"id": v[0], "label": self.get_label(v[0])} for v in vals],
                        "is_own":  attr_id in own_attr_ids,
                    })

            # ── 5. Example children ───────────────────────────────────────────
            example_children = []
            for (cid,) in child_rows[:6]:
                child_roles = c.execute(
                    "SELECT attr_id, val_id FROM roles WHERE concept_id=?", (cid,)
                ).fetchall()
                parts = [
                    f"{r[0]} | {self.get_label(r[0])} | = "
                    f"{r[1]} | {self.get_label(r[1])} |"
                    for r in child_roles
                ]
                expr = f"{snomed_code} | {focus_label} |"
                if parts:
                    expr += " :\n  { " + ",\n    ".join(parts) + " }"
                example_children.append({
                    "id":         cid,
                    "label":      self.get_label(cid),
                    "expression": expr,
                })

            return {
                "focus":           {"id": snomed_code, "label": focus_label},
                "parents":         parents,
                "own_roles":       own_roles,
                "own_expression":  own_expression,
                "attributes":      attributes,
                "example_children": example_children,
            }
