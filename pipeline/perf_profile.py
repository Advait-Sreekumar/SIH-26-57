"""Performance profiling for all 11 pipeline stages.

Run from repo root or pipeline/:
    python pipeline/perf_profile.py

Hardware: Intel i7-12700H + RTX 4050 Laptop GPU, 16 GB RAM
All times via time.perf_counter() (wall-clock, single run).
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

results = []

def rec(stage, t, notes=""):
    results.append((stage, t, notes))
    print(f"  {stage:<52s}  {t*1000:>8.1f} ms  {notes}")

print("")
print("=== SonarEye Performance Profile ===")
print("  Hardware: Intel i7-12700H + RTX 4050 Laptop GPU, 16 GB RAM")
print("")
