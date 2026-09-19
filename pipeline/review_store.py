import sqlite3
import datetime
from pathlib import Path

VALID_ACTIONS = {"confirm", "reject", "uncertain", "annotate"}
_NOTE_MAX = 500


def get_db_path() -> Path:
    return Path(__file__).parent / "review_store.db"


VALID_SOURCES = {"sample", "upload"}
_SAMPLE_MAX_AGE_HOURS = 24


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Add source column if missing. Existing rows default to 'upload' (safe)."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    if "source" not in cols:
        conn.execute(
            "ALTER TABLE runs ADD COLUMN source TEXT NOT NULL DEFAULT 'upload'"
        )
        conn.commit()


def init_db(db_path=None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else get_db_path()
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            image_name TEXT NOT NULL,
            n_detections INTEGER NOT NULL,
            source TEXT NOT NULL DEFAULT 'upload'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            detection_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            category TEXT,
            note TEXT,
            reviewed_at TEXT NOT NULL,
            UNIQUE(run_id, detection_id)
        )
    """)
    _ensure_schema(conn)
    conn.commit()
    return conn


def register_run(
    conn: sqlite3.Connection,
    run_id: str,
    image_name: str,
    n_detections: int,
    source: str = "upload",
):
    if source not in VALID_SOURCES:
        raise ValueError(f"Invalid source {source!r}. Must be one of {sorted(VALID_SOURCES)}")
    _ensure_schema(conn)
    conn.execute(
        "INSERT OR IGNORE INTO runs (run_id, created_at, image_name, n_detections, source) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, _utcnow(), image_name, n_detections, source),
    )
    conn.commit()


def get_run_source(conn: sqlite3.Connection, run_id: str):
    _ensure_schema(conn)
    row = conn.execute("SELECT source FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return row[0] if row else None


def delete_sample_run(conn: sqlite3.Connection, run_id: str) -> bool:
    """Hard-delete reviews + run row only when source is 'sample'. Returns True if deleted."""
    _ensure_schema(conn)
    src = get_run_source(conn, run_id)
    if src != "sample":
        return False
    conn.execute("DELETE FROM reviews WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
    conn.commit()
    return True


def purge_old_sample_runs(
    conn: sqlite3.Connection,
    max_age_hours: int = _SAMPLE_MAX_AGE_HOURS,
    keep_run_id: str | None = None,
) -> int:
    """Belt-and-suspenders: drop sample runs older than max_age_hours. Never touches uploads."""
    _ensure_schema(conn)
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=max_age_hours)
    rows = conn.execute(
        "SELECT run_id, created_at FROM runs WHERE source = 'sample'"
    ).fetchall()
    n = 0
    for rid, created in rows:
        if keep_run_id and rid == keep_run_id:
            continue
        try:
            ts = datetime.datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
        if ts < cutoff:
            if delete_sample_run(conn, rid):
                n += 1
    return n


def upsert_review(
    conn: sqlite3.Connection,
    run_id: str,
    detection_id: int,
    action: str,
    category=None,
    note=None,
):
    if action not in VALID_ACTIONS:
        raise ValueError(f"Invalid action {action!r}. Must be one of {sorted(VALID_ACTIONS)}")
    note = _clean_note(note)
    conn.execute(
        """
        INSERT INTO reviews (run_id, detection_id, action, category, note, reviewed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, detection_id) DO UPDATE SET
            action = excluded.action,
            category = excluded.category,
            note = excluded.note,
            reviewed_at = excluded.reviewed_at
        """,
        (run_id, detection_id, action, category or None, note, _utcnow()),
    )
    conn.commit()


def get_reviews_for_run(conn: sqlite3.Connection, run_id: str) -> dict:
    cur = conn.execute(
        "SELECT detection_id, action, category, note, reviewed_at FROM reviews WHERE run_id = ?",
        (run_id,),
    )
    return {
        row[0]: {
            "action": row[1],
            "category": row[2],
            "note": row[3],
            "reviewed_at": row[4],
        }
        for row in cur.fetchall()
    }


def delete_review(conn: sqlite3.Connection, run_id: str, detection_id: int):
    conn.execute(
        "DELETE FROM reviews WHERE run_id = ? AND detection_id = ?",
        (run_id, detection_id),
    )
    conn.commit()


def _utcnow() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean_note(note) -> object:
    if not note:
        return None
    note = note.strip()
    if not note:
        return None
    return note[:_NOTE_MAX]
