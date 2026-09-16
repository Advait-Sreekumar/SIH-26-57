import sqlite3
import tempfile
from pathlib import Path
import pytest

from review_store import (
    VALID_ACTIONS,
    delete_review,
    get_reviews_for_run,
    init_db,
    register_run,
    upsert_review,
)


def _make_db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    return conn, db_path


def test_persistence_across_restart(tmp_path):
    conn1, db_path = _make_db(tmp_path)
    register_run(conn1, "run1", "img.png", 3)
    upsert_review(conn1, "run1", 0, "confirm", "shipwreck", "looks good")
    conn1.close()

    conn2 = init_db(db_path)
    reviews = get_reviews_for_run(conn2, "run1")
    assert 0 in reviews
    assert reviews[0]["action"] == "confirm"
    assert reviews[0]["category"] == "shipwreck"
    conn2.close()


def test_reviews_isolated_by_run_id(tmp_path):
    conn, _ = _make_db(tmp_path)
    register_run(conn, "run_a", "a.png", 1)
    register_run(conn, "run_b", "b.png", 1)
    upsert_review(conn, "run_a", 0, "confirm")
    revs_b = get_reviews_for_run(conn, "run_b")
    assert revs_b == {}
    conn.close()


def test_invalid_action_raises(tmp_path):
    conn, _ = _make_db(tmp_path)
    register_run(conn, "run1", "img.png", 1)
    with pytest.raises(ValueError):
        upsert_review(conn, "run1", 0, "approve")
    conn.close()


def test_foreign_key_enforced(tmp_path):
    conn, _ = _make_db(tmp_path)
    # run_id "ghost" was never registered
    with pytest.raises(sqlite3.IntegrityError):
        upsert_review(conn, "ghost", 0, "confirm")
    conn.close()


def test_empty_note_stored_as_none(tmp_path):
    conn, _ = _make_db(tmp_path)
    register_run(conn, "run1", "img.png", 1)
    upsert_review(conn, "run1", 0, "confirm", note="")
    rev = get_reviews_for_run(conn, "run1")
    assert rev[0]["note"] is None
    conn.close()


def test_overwrite_review(tmp_path):
    conn, _ = _make_db(tmp_path)
    register_run(conn, "run1", "img.png", 1)
    upsert_review(conn, "run1", 0, "confirm", "shipwreck", "first")
    upsert_review(conn, "run1", 0, "reject", "rock", "second")
    rev = get_reviews_for_run(conn, "run1")
    assert rev[0]["action"] == "reject"
    assert rev[0]["note"] == "second"
    # only one row
    cur = conn.execute("SELECT count(*) FROM reviews WHERE run_id='run1'")
    assert cur.fetchone()[0] == 1
    conn.close()


def test_long_note_truncated(tmp_path):
    conn, _ = _make_db(tmp_path)
    register_run(conn, "run1", "img.png", 1)
    long_note = "x" * 1000
    upsert_review(conn, "run1", 0, "annotate", note=long_note)
    rev = get_reviews_for_run(conn, "run1")
    assert len(rev[0]["note"]) == 500
    conn.close()
