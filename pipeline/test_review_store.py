import sqlite3
import tempfile
from pathlib import Path
import pytest

from review_store import (
    VALID_ACTIONS,
    delete_review,
    delete_sample_run,
    get_reviews_for_run,
    get_run_source,
    init_db,
    purge_old_sample_runs,
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


def test_delete_sample_run_only_when_tagged_sample(tmp_path):
    conn, _ = _make_db(tmp_path)
    register_run(conn, "demo1", "Corsair_01.png", 1, source="sample")
    upsert_review(conn, "demo1", 0, "confirm", note="demo")
    register_run(conn, "real1", "field.png", 1, source="upload")
    upsert_review(conn, "real1", 0, "reject", note="keep me")

    assert delete_sample_run(conn, "demo1") is True
    assert get_reviews_for_run(conn, "demo1") == {}
    assert get_run_source(conn, "demo1") is None

    assert delete_sample_run(conn, "real1") is False
    assert get_reviews_for_run(conn, "real1")[0]["action"] == "reject"
    assert get_run_source(conn, "real1") == "upload"
    conn.close()


def test_untagged_legacy_run_defaults_to_upload_and_is_not_deleted(tmp_path):
    conn, db_path = _make_db(tmp_path)
    conn.execute(
        "INSERT INTO runs (run_id, created_at, image_name, n_detections) VALUES (?,?,?,?)",
        ("legacy", "2020-01-01T00:00:00Z", "old.png", 1),
    )
    conn.commit()
    conn.close()
    conn = init_db(db_path)
    assert get_run_source(conn, "legacy") == "upload"
    assert delete_sample_run(conn, "legacy") is False
    conn.close()


def test_purge_old_sample_runs_spares_uploads_and_fresh_samples(tmp_path):
    conn, _ = _make_db(tmp_path)
    conn.execute(
        "INSERT INTO runs (run_id, created_at, image_name, n_detections, source) VALUES (?,?,?,?,?)",
        ("old_s", "2020-01-01T00:00:00Z", "a.png", 1, "sample"),
    )
    conn.execute(
        "INSERT INTO runs (run_id, created_at, image_name, n_detections, source) VALUES (?,?,?,?,?)",
        ("old_u", "2020-01-01T00:00:00Z", "b.png", 1, "upload"),
    )
    register_run(conn, "new_s", "c.png", 1, source="sample")
    conn.commit()
    n = purge_old_sample_runs(conn, max_age_hours=24, keep_run_id="new_s")
    assert n == 1
    assert get_run_source(conn, "old_s") is None
    assert get_run_source(conn, "old_u") == "upload"
    assert get_run_source(conn, "new_s") == "sample"
    conn.close()
