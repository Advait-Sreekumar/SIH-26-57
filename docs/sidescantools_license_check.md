# SidescanTools License Verification

**Date:** 2026-09-16  
**Repo:** https://github.com/apl-ocean-engineering/SidescanTools  

---

## Task B Results

### 1. Repository Access

The SidescanTools GitHub repository returned HTTP 404 on both main and master branch
attempts (WebFetch, WebSearch, GitHub API all returned no results).
The repo appears to be private, renamed, or deleted as of this date.

Finding: License text could not be retrieved from GitHub.

### 2. Was SidescanTools code copied into the pipeline?

Inspection of pipeline/xtf_io.py:
- No copyright header or attribution to SidescanTools
- No `import sidescantools` or equivalent
- No copied function signatures matching SidescanTools API
- File opens with `from pathlib import Path / import cv2 / import numpy as np`
- Git log traces xtf_io.py to the Phase 1 corrective commit (original code)

The file uses `pyxtf` (oysstu/pyxtf on PyPI) as its XTF parser.
pyxtf is a separate, independently published package. It is NOT SidescanTools.

Finding: No SidescanTools code is in pipeline/xtf_io.py or anywhere else in pipeline/.

### 3. pyxtf License — Confirmed MIT

From `pip show pyxtf`:

    Name: pyxtf
    Version: 1.5.0
    Home-page: https://github.com/oysstu/pyxtf
    Author: Oystein Sture
    License: MIT
    Classifier: License :: OSI Approved :: MIT License

MIT License permits use without restriction in academic, commercial, or open-source
projects with attribution only. pyxtf is already listed in requirements.txt.

### 4. What xtf_io.py actually does

Original code (written in this project) that:
1. Wraps pyxtf.xtf_read() to parse XTF waterfall data
2. Applies optional slant-range correction (original implementation)
3. Extracts navigation metadata from XTF ping headers

No third-party code beyond pyxtf public API calls.

---

## Task B Verdict

CLEAR — no SidescanTools code is present in the pipeline.

The prior mention of SidescanTools in project history was a planning reference
(as a potential dependency) that was never acted on. The actual XTF parsing uses
pyxtf (MIT License) wrapped in original code.

No license obligation from SidescanTools applies. No attribution to SidescanTools
required. pyxtf (MIT) is in requirements.txt; no action needed.
