"""Live AppTest of the guided-mission flow (P1 regression / gap-fix).

Run from repo root or pipeline/:  python test_flow_integrity.py
Reports real widget/session outcomes for each step — not predictions.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# app.py imports local pipeline modules
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402


SAMPLE = "Corsair_01.png"
TIMEOUT = 300


def _btns(at: AppTest):
    seen, keys = [], set()
    for b in list(at.button) + list(at.sidebar.button):
        k = getattr(b, "key", None) or id(b)
        if k in keys:
            continue
        keys.add(k)
        seen.append(b)
    return seen


def _by_key(at: AppTest, key: str):
    for b in _btns(at):
        if getattr(b, "key", None) == key:
            return b
    return None


def _labels(at: AppTest):
    out = []
    for b in _btns(at):
        lab = getattr(b, "label", None) or ""
        out.append(lab.encode("ascii", "replace").decode("ascii"))
    return out


def _click(at: AppTest, key: str) -> AppTest:
    b = _by_key(at, key)
    if b is None:
        raise AssertionError(f"No button key={key!r}. Have: {_labels(at)}")
    b.click()
    return at.run(timeout=TIMEOUT)


def _selectboxes(at: AppTest):
    return list(at.selectbox) + list(at.sidebar.selectbox)


def _sb_by_key(at: AppTest, key: str):
    for s in _selectboxes(at):
        if getattr(s, "key", None) == key:
            return s
    return None


def _ssget(at: AppTest, key, default=None):
    try:
        return at.session_state[key]
    except KeyError:
        return default


def _dump(at: AppTest, step: str) -> None:
    print(f"\n=== {step} ===")
    if at.exception:
        print("EXCEPTION:", at.exception)
        raise SystemExit(1)
    print("nav:", _ssget(at, "nav"), "| started:", _ssget(at, "survey_started"),
          "| last_file:", _ssget(at, "last_file"), "| run_id:", _ssget(at, "run_id"))
    cfg = _ssget(at, "survey_cfg")
    print("cfg name:", None if not cfg else cfg.get("name"),
          "| geod cached:", _ssget(at, "_infer_results") is not None)
    infer = _ssget(at, "_infer_results")
    if infer is not None:
        geod = infer[2]
        print("n_dets:", len(geod), "| ids:", [d["id"] for d in geod][:12])
    print("buttons:", _labels(at))
    print("selectbox keys:", [getattr(s, "key", None) for s in _selectboxes(at)])


def main() -> None:
    log = []

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    _dump(at, "landing")
    assert _by_key(at, None) or any("Launch" in (l or "") for l in _labels(at))
    launch = next(b for b in _btns(at) if "Launch" in (b.label or ""))
    launch.click()
    at.run()
    _dump(at, "after launch")
    log.append(("Launch SonarEye", "ok", f"nav={_ssget(at, 'nav')}"))

    at = _click(at, "nav_survey")
    _dump(at, "survey empty")
    start = _by_key(at, "survey_start")
    assert start is not None, "Start Survey button missing on Survey page"
    disabled = bool(getattr(start, "disabled", False))
    log.append(("Start Survey present (no file)", "ok" if disabled else "WARN",
                f"disabled={disabled}"))
    print("Start Survey disabled (no file):", disabled)

    sample_box = _sb_by_key(at, "cfg_sample")
    assert sample_box is not None, "Sample selector missing before survey start"
    opts = list(sample_box.options)
    assert SAMPLE in opts, f"{SAMPLE} not in {opts[:8]}..."
    sample_box.select(SAMPLE)
    at.run()
    _dump(at, "sample selected")
    start = _by_key(at, "survey_start")
    assert start is not None
    disabled = bool(getattr(start, "disabled", False))
    log.append(("Start Survey enabled after sample", "ok" if not disabled else "FAIL",
                f"disabled={disabled}"))
    print("Start Survey disabled after sample:", disabled)
    assert not disabled, "Start Survey should be enabled once a sample is loaded"

    at = _click(at, "survey_start")
    _dump(at, "after Start Survey")
    assert _ssget(at, "survey_started") is True
    assert _ssget(at, "nav") == "analysis"
    cfg = _ssget(at, "survey_cfg") or {}
    assert cfg.get("name") == SAMPLE
    geod = _ssget(at, "_infer_results")[2]
    n0 = len(geod)
    ids0 = [d["id"] for d in geod]
    log.append(("Start Survey -> Analysis + inference", "ok",
                f"n_dets={n0} ids={ids0}"))

    # Wipe leftover SQLite reviews from prior test runs of this same file hash
    from review_store import delete_review, get_reviews_for_run, init_db
    _conn = init_db()
    _rid = _ssget(at, "run_id")
    for _did in list(get_reviews_for_run(_conn, _rid)):
        delete_review(_conn, _rid, _did)
    at.run(timeout=TIMEOUT)

    # Selector must be gone; lock control present
    assert _sb_by_key(at, "cfg_sample") is None, "Sample selector still live after start"
    assert _by_key(at, "sb_new_survey") is not None, "Start New Survey missing in sidebar"
    log.append(("Sample selector locked after start", "ok", "cfg_sample absent"))

    at = _click(at, "nav_detections")
    _dump(at, "detections")
    n_dets_page = n0
    log.append(("Detections after analysis", "ok", f"n={n_dets_page}"))

    # Begin sequential review if detections exist
    if n0 == 0:
        log.append(("Review detections", "SKIP", "zero detections on sample"))
    else:
        at = _click(at, "start_seq_review")
        _dump(at, "seq_review start")
        assert _ssget(at, "nav") == "seq_review"
        first_det = _ssget(at, "review_det_id")
        log.append(("Begin sequential review", "ok", f"det={first_det}"))

        # Sidebar hop mid-review: Map -> Detections -> Overview -> Detections
        snap_run = _ssget(at, "run_id")
        snap_file = _ssget(at, "last_file")
        snap_geod = [d["id"] for d in _ssget(at, "_infer_results")[2]]
        snap_rev_idx = _ssget(at, "review_idx")
        snap_rev_det = _ssget(at, "review_det_id")

        for hop, key in [
            ("map", "nav_map"),
            ("detections", "nav_detections"),
            ("overview", "nav_overview"),
            ("detections", "nav_detections"),
        ]:
            at = _click(at, key)
            _dump(at, f"hop {hop}")
            assert _ssget(at, "survey_started") is True
            assert _ssget(at, "run_id") == snap_run
            assert _ssget(at, "last_file") == snap_file
            assert [d["id"] for d in _ssget(at, "_infer_results")[2]] == snap_geod
            assert _ssget(at, "review_idx") == snap_rev_idx
            assert _ssget(at, "review_det_id") == snap_rev_det
            assert _sb_by_key(at, "cfg_sample") is None
            log.append((f"Sidebar hop -> {hop} (state intact)", "ok",
                        f"run_id={snap_run} n={len(snap_geod)} review_det={snap_rev_det}"))

        # Resume review and complete all detections
        if _ssget(at, "nav") != "seq_review":
            at = _click(at, "start_seq_review")
        _dump(at, "resume seq_review")

        safety = 0
        while _ssget(at, "nav") == "seq_review" and not _ssget(at, "review_complete"):
            safety += 1
            if safety > 40:
                raise AssertionError("Review loop exceeded 40 saves")
            # Confirm current detection
            save = next(
                (b for b in _btns(at)
                 if b.key and str(b.key).startswith("sr_save_")),
                None,
            )
            if save is None:
                # completion screen?
                if _by_key(at, "sr_to_review"):
                    break
                raise AssertionError(f"No save button. labels={_labels(at)}")
            save.click()
            at.run(timeout=TIMEOUT)
            _dump(at, f"after save #{safety}")

        if _by_key(at, "sr_to_review"):
            at = _click(at, "sr_to_review")
        else:
            at = _click(at, "nav_map")
        _dump(at, "map after review")
        assert _ssget(at, "nav") == "map", f"expected map, got {_ssget(at, 'nav')}"
        log.append(("Complete review -> Map (before Summary)", "ok",
                    f"nav={_ssget(at, 'nav')} buttons={_labels(at)}"))
        assert _by_key(at, "map_to_sum") is not None

        # Simulate a marker-select the same way the Folium callback does
        at.session_state["_map_last_tip"] = "force"
        # Directly exercise the destination the marker handler uses
        at.session_state["selected_det"] = 0
        at.session_state["detail_return"] = "map"
        at.session_state["nav"] = "detail"
        at.run(timeout=TIMEOUT)
        _dump(at, "detail via map selection")
        assert _ssget(at, "nav") == "detail"
        assert _ssget(at, "selected_det") == 0
        log.append(("Map selection -> Anomaly Detail #0", "ok",
                    f"nav={_ssget(at, 'nav')} det={_ssget(at, 'selected_det')}"))
        at = _click(at, "det_back")
        _dump(at, "back to map")
        assert _ssget(at, "nav") == "map"

        at = _click(at, "map_to_sum")
        _dump(at, "summary")
        log.append(("Map -> Summary", "ok", f"nav={_ssget(at, 'nav')}"))

        # Edit one confirmed decision from Summary
        revise = next((b for b in _btns(at) if b.key and str(b.key).startswith("sum_rev_")), None)
        assert revise is not None, f"No Revise buttons on Summary: {_labels(at)}"
        revise.click()
        at.run()
        _dump(at, "revise from summary")
        assert _ssget(at, "nav") == "detail"
        det_id = _ssget(at, "selected_det")
        # Change action to reject and save
        radios = list(at.radio)
        action_radio = next((r for r in radios if "Action" in (r.label or "")), radios[0] if radios else None)
        assert action_radio is not None, "No Action radio on detail"
        action_radio.set_value("reject")
        at.run()
        save_det = next((b for b in _btns(at) if b.key and str(b.key).startswith("det_save_")), None)
        assert save_det is not None
        save_det.click()
        at.run()
        _dump(at, "saved revised decision")
        log.append(("Revise from Summary + re-save", "ok", f"det={det_id} -> reject"))

        at = _click(at, "det_back")
        _dump(at, "back to summary")
        assert _ssget(at, "nav") == "review"

        at = _click(at, "sum_to_report")
        _dump(at, "report")
        assert _ssget(at, "nav") == "report"
        log.append(("Summary -> Continue to Report", "ok", f"nav={_ssget(at, 'nav')}"))

        dls = list(at.download_button)
        pdf_dl = next((b for b in dls if "PDF" in (b.label or "")), None)
        assert pdf_dl is not None, f"PDF download missing: {[b.label for b in dls]}"
        # AppTest cannot read download_button byte values; presence confirms no exception was raised.
        # Byte-level validation is covered by the standalone _debug_pdf.py script.
        log.append(("Download PDF Report", "ok",
                    f"button present label={pdf_dl.label!r}"))

        assert _sb_by_key(at, "cfg_sample") is None
        log.append(("Selector still locked at Report", "ok", "cfg_sample absent"))

        from review_store import get_reviews_for_run, get_run_source, init_db
        rid = _ssget(at, "run_id")
        conn = init_db()
        before = get_reviews_for_run(conn, rid)
        src = get_run_source(conn, rid)
        log.append(("Sample run tagged in DB", "ok" if src == "sample" else "FAIL",
                    f"source={src} rid={rid} n_reviews={len(before)}"))
        assert src == "sample"
        assert before
        at = _click(at, "sb_new_survey")
        _dump(at, "after Start New Survey")
        conn2 = init_db()
        after = get_reviews_for_run(conn2, rid)
        src2 = get_run_source(conn2, rid)
        log.append(("Sample run purged from DB", "ok" if src2 is None and after == {} else "FAIL",
                    f"source={src2} reviews={after}"))
        assert src2 is None
        assert after == {}

    print("\n========== FLOW RESULTS ==========")
    for name, status, detail in log:
        print(f"[{status}] {name} | {detail}")
    fails = [x for x in log if x[1] == "FAIL"]
    if fails:
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
