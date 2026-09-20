"""Generate all Part B diagrams for docs/diagrams/.

Tool: matplotlib (no graphviz required).
Run from repo root:  python docs/diagrams/generate_diagrams.py
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
OUT  = Path(__file__).resolve().parent                # docs/diagrams/

# shared style
DARK_BG  = "#1a1a2e"
ACCENT   = "#e94560"
MID_BLUE = "#0f3460"
LIGHT    = "#e0e0e0"
GREEN    = "#2e7d32"
GREEN_L  = "#4caf50"
AMBER    = "#ff9800"
RED_SOFT = "#ef5350"
GREY     = "#78909c"
TEAL     = "#00695c"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   DARK_BG,
    "text.color":       LIGHT,
    "axes.labelcolor":  LIGHT,
    "xtick.color":      LIGHT,
    "ytick.color":      LIGHT,
    "axes.edgecolor":   GREY,
    "grid.color":       "#2a2a4a",
    "font.family":      "DejaVu Sans",
    "font.size":        10,
})


def _box(ax, cx, cy, w, h, label, sublabel="", color=MID_BLUE, textcolor=LIGHT,
         fontsize=9, edge=ACCENT):
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.04", linewidth=1.4,
        edgecolor=edge, facecolor=color, zorder=3,
    )
    ax.add_patch(rect)
    y_label = cy + (0.10 if sublabel else 0)
    ax.text(cx, y_label, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=textcolor, zorder=4)
    if sublabel:
        ax.text(cx, cy - 0.18, sublabel, ha="center", va="center",
                fontsize=7.5, color="#aaaaaa", zorder=4, style="italic")


def _arrow(ax, x0, y0, x1, y1, color=ACCENT, lw=1.8):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, mutation_scale=14),
                zorder=5)


# ===========================================================================
# 1. System architecture / data-flow
# ===========================================================================
def diagram_system_architecture():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    ax.text(7, 6.65, "SonarEye - Pipeline Data Flow",
            ha="center", va="center", fontsize=13, fontweight="bold", color=LIGHT)
    ax.text(7, 6.30, "Matches MASTER_DOCUMENTATION.md section 3",
            ha="center", va="center", fontsize=8, color=GREY)

    stages = [
        (1.1,  3.8, "Raw Sonar",       "XTF / PNG / JPEG"),
        (3.0,  3.8, "Preprocessing",   "Lee filter + water crop\nnormalise + CLAHE"),
        (5.2,  3.8, "Tiled Inference", "U-Net ResNet34\n512x512 tiles + blend"),
        (7.4,  3.8, "Post-processing", "Connected components\nConfidence scoring"),
        (9.6,  3.8, "Geotagging",      "pixel to lat/lon\nPixelGeoMapper"),
        (11.8, 3.8, "Mission Report",  "JSON + CSV + PDF"),
    ]
    for (cx, cy, lbl, sub) in stages:
        _box(ax, cx, cy, 1.6, 1.0, lbl, sub, fontsize=8.5)

    xs = [s[0] for s in stages]
    for i in range(len(xs) - 1):
        _arrow(ax, xs[i] + 0.8, 3.8, xs[i+1] - 0.8, 3.8)

    loop_cx = 9.1
    loop_cy = 1.7
    _box(ax, loop_cx, loop_cy, 3.8, 1.1,
         "Human Review Loop",
         "confirm / reject / uncertain\nSQLite persistence (review_store.py)",
         color="#1b3a3a", edge=TEAL, fontsize=8.5)

    _arrow(ax, 7.4, 3.3, 7.9, loop_cy + 0.55, color=TEAL)
    ax.text(7.05, 2.6, "detections", fontsize=7.5, color=TEAL, ha="center")

    _arrow(ax, 9.6, 3.3, 9.4, loop_cy + 0.55, color=TEAL)
    ax.text(9.9, 2.7, "geo-tagged", fontsize=7.5, color=TEAL, ha="center")

    _arrow(ax, loop_cx + 1.9, loop_cy, 11.8 - 0.8, 3.3, color=TEAL)
    ax.text(11.2, 2.5, "reviewed\ndecisions", fontsize=7.5, color=TEAL, ha="center")

    patches = [
        mpatches.Patch(facecolor=MID_BLUE,  edgecolor=ACCENT, label="Automated pipeline stage"),
        mpatches.Patch(facecolor="#1b3a3a", edgecolor=TEAL,   label="Human-in-the-loop stage"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=8,
              facecolor=DARK_BG, edgecolor=GREY, labelcolor=LIGHT)

    path = OUT / "01_system_architecture.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [1] Saved: {path.name}")


# ===========================================================================
# 2. Confidence scoring formula breakdown
# Source: pipeline/confidence.py line 76
# conf = 100 * (0.6 * model_score + 0.4 * (0.7 * geo + 0.3 * shadow))
# ===========================================================================
def diagram_confidence_score():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    ax.text(6.5, 6.2, "Confidence Score - Formula Breakdown  (confidence.py:76)",
            ha="center", va="center", fontsize=12, fontweight="bold", color=LIGHT)
    ax.text(6.5, 5.82, "Heuristic composite - not a calibrated probability",
            ha="center", va="center", fontsize=9, color=AMBER, style="italic")

    _box(ax, 6.5, 4.9, 4.0, 0.7, "Confidence Score  (0-100)",
         "100 x final_composite", color="#3a1a2e", edge=ACCENT, fontsize=9)

    _box(ax, 3.0, 3.5, 3.2, 0.75, "Model Score  x0.60",
         "0.5 x mean_prob + 0.5 x peak_prob", color="#1a2e3a", edge="#42a5f5", fontsize=8.5)
    _box(ax, 9.8, 3.5, 3.4, 0.75, "Heuristic Score  x0.40",
         "0.7 x geo + 0.3 x shadow", color="#1a2e1a", edge="#66bb6a", fontsize=8.5)

    _arrow(ax, 3.0,  3.87, 5.5, 4.55, color="#42a5f5")
    _arrow(ax, 9.8,  3.87, 7.5, 4.55, color="#66bb6a")

    _box(ax, 1.5, 2.1, 2.2, 0.7, "mean_prob  x0.5",
         "mean sigmoid in bbox", color="#0d1b2a", edge="#42a5f5", fontsize=8)
    _box(ax, 4.0, 2.1, 2.2, 0.7, "peak_prob  x0.5",
         "max sigmoid in bbox", color="#0d1b2a", edge="#42a5f5", fontsize=8)
    _arrow(ax, 1.5, 2.45, 2.4, 3.13, color="#42a5f5", lw=1.3)
    _arrow(ax, 4.0, 2.45, 3.6, 3.13, color="#42a5f5", lw=1.3)

    _box(ax, 7.9, 2.1, 2.0, 0.7, "Geo  x0.70",
         "0.40 x aspect + 0.25 x solidity\n+ 0.35 x edge_sharpness",
         color="#0d2a0d", edge="#66bb6a", fontsize=8)
    _box(ax, 11.3, 2.1, 2.0, 0.7, "Shadow  x0.30",
         "target-ring contrast\n(clipped 0-1)",
         color="#0d2a0d", edge="#66bb6a", fontsize=8)
    _arrow(ax, 7.9,  2.45, 9.1, 3.13, color="#66bb6a", lw=1.3)
    _arrow(ax, 11.3, 2.45, 10.5, 3.13, color="#66bb6a", lw=1.3)

    geo_items = [
        (6.5,  0.95, "Aspect\nx0.40",         "major/minor axis\nideal ~3.5"),
        (8.5,  0.95, "Solidity\nx0.25",        "area/hull area\n(hull fill ratio)"),
        (10.5, 0.95, "Edge Sharpness\nx0.35",  "boundary grad /\ninterior grad"),
    ]
    for (cx, cy, lbl, sub) in geo_items:
        _box(ax, cx, cy, 1.7, 0.75, lbl, sub,
             color="#0a1a0a", edge="#388e3c", fontsize=7.5)
        _arrow(ax, cx, cy + 0.38, 7.9, 2.1 - 0.35, color="#388e3c", lw=1.1)

    ax.text(6.5, 0.18,
            "Weights are hand-tuned. No Platt scaling or isotonic regression. "
            "Score is useful as a relative within-survey ranking only.",
            ha="center", va="center", fontsize=7.5, color=AMBER, style="italic")

    path = OUT / "02_confidence_score_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [2] Saved: {path.name}")


# ===========================================================================
# 3. Evaluation summary charts
# Source A: evaluation_report.md section 1 (IoU figures)
# Source B: pipeline/robustness_results.json (perturbation deltas)
# ===========================================================================
def diagram_evaluation_charts():
    rob_path = ROOT / "pipeline" / "robustness_results.json"
    with open(rob_path) as f:
        rob = json.load(f)

    baseline_iou = rob["baseline"]["mean_iou"]  # 0.0581
    perturbs = rob["perturbations"]
    pert_keys = ["speckle_noise", "synthesize_shadow", "radial_distortion", "heave_pitch_roll"]
    pert_labels = ["Speckle Noise", "Shadow Synth", "Radial Distort", "Heave/Pitch/Roll"]
    pert_deltas = [perturbs[k]["iou_delta_vs_baseline"] for k in pert_keys]
    pert_ious   = [perturbs[k]["mean_iou"] for k in pert_keys]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("SonarEye - Evaluation Summary", fontsize=12,
                 fontweight="bold", color=LIGHT, y=1.01)

    # Panel A: Test vs Validation IoU
    # Source: evaluation_report.md section 1
    # Test IoU = 0.4268 (13 held-out sites, primary metric)
    # Val  IoU = 0.713  (12 train sites, in-distribution)
    ax = axes[0]
    ax.set_facecolor(DARK_BG)
    labels = ["Validation IoU\n(in-distribution,\n12 train sites)",
              "Test IoU\n(held-out,\n13 unseen sites)"]
    values = [0.713, 0.4268]
    colors = [GREY, ACCENT]
    bars = ax.bar(labels, values, color=colors, edgecolor="#444", width=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean pixel IoU", color=LIGHT)
    ax.set_title("A. Segmentation IoU\n(primary metric = Test)",
                 color=LIGHT, fontsize=10)
    ax.axhline(0.55, color=AMBER, lw=1.2, ls="--")
    ax.axhline(0.77, color=AMBER, lw=1.2, ls="--")
    ax.text(1.42, 0.59, "SOTA range\n0.55-0.77", fontsize=7.5, color=AMBER)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.018,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=LIGHT)
    ax.text(0.5, -0.14,
            "Source: evaluation_report.md section 1\nTest IoU is the authoritative number",
            ha="center", transform=ax.transAxes, fontsize=7.5, color=GREY, style="italic")

    # Panel B: Robustness perturbation deltas
    # Source: pipeline/robustness_results.json (loaded above)
    ax2 = axes[1]
    ax2.set_facecolor(DARK_BG)
    bar_colors = [RED_SOFT if d < -0.01 else AMBER for d in pert_deltas]
    bars2 = ax2.barh(pert_labels, pert_deltas, color=bar_colors, edgecolor="#444", height=0.55)
    ax2.set_xlim(-0.055, 0.005)
    ax2.axvline(0, color=GREY, lw=0.9)
    ax2.set_xlabel("IoU delta vs clean baseline", color=LIGHT)
    ax2.set_title(f"B. Robustness - IoU delta per perturbation\n"
                  f"(baseline IoU = {baseline_iou:.4f})",
                  color=LIGHT, fontsize=10)
    for bar, delta in zip(bars2, pert_deltas):
        ax2.text(delta - 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{delta:+.4f}",
                 ha="right", va="center", fontsize=8.5, color=LIGHT, fontweight="bold")
    ax2.text(0.5, -0.14,
             f"Source: pipeline/robustness_results.json\n"
             f"Fields: perturbations[*].iou_delta_vs_baseline, baseline.mean_iou",
             ha="center", transform=ax2.transAxes, fontsize=7.5, color=GREY, style="italic")

    fig.tight_layout()
    path = OUT / "03_evaluation_charts.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [3] Saved: {path.name}")
    print(f"       IoU source: evaluation_report.md section 1 (test=0.4268, val=0.713)")
    print(f"       Robustness source: {rob_path}")
    print(f"         fields: baseline.mean_iou, perturbations[*].iou_delta_vs_baseline")


# ===========================================================================
# 4. Class-expansion decision diagram
# ===========================================================================
def diagram_class_expansion():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    ax.text(7, 6.65, "Class Expansion - Investigation Outcomes",
            ha="center", va="center", fontsize=13, fontweight="bold", color=LIGHT)
    ax.text(7, 6.32, "Each branch shows what was investigated and why it was closed or deferred",
            ha="center", va="center", fontsize=9, color=GREY)

    _box(ax, 7, 5.4, 3.2, 0.75, "Expand detection classes",
         "SIH brief requirement", color="#2a1a3a", edge=ACCENT, fontsize=9)

    bx = [1.8, 5.2, 8.8, 12.2]
    branch_y = 3.8

    configs = [
        (bx[0], branch_y, "Shipwrecks",    "SHIPPED",
         "AI4Shipwrecks dataset\n286 images, 28 sites\nTest IoU 0.427",
         GREEN, GREEN_L),
        (bx[1], branch_y, "Pipes + Cylinders", "BLOCKED",
         "SubPipe dataset: GPL-3.0\nlicense blocks commercial use.\nAUV-towfish domain shift.",
         "#3a2000", AMBER),
        (bx[2], branch_y, "Ghost Nets",    "BLOCKED",
         "No public labelled sonar\ndata exists. Synthetic-only\nexperiment - unvalidated.",
         "#3a1a00", RED_SOFT),
        (bx[3], branch_y, "Generic Debris", "DEFERRED",
         "No labelled dataset found.\nSynthetic-only exploration.\nUnproven capability.",
         "#2a2a1a", GREY),
    ]

    status_colors = {"SHIPPED": GREEN_L, "BLOCKED": RED_SOFT, "DEFERRED": GREY}

    for (cx, cy, title, status, detail, bg, edge) in configs:
        _box(ax, cx, cy, 2.9, 1.1, title, detail, color=bg, edge=edge, fontsize=8.5)
        badge_color = status_colors[status]
        ax.text(cx, cy + 0.9, status, ha="center", va="center",
                fontsize=8, fontweight="bold", color=DARK_BG,
                bbox=dict(facecolor=badge_color, edgecolor="none",
                          boxstyle="round,pad=0.25"),
                zorder=6)
        _arrow(ax, 7, 5.03, cx, cy + 0.55, color=edge, lw=1.5)

    blockers = [
        (bx[1], 1.9,
         "License: GPL-3.0\n-> Investigated; cannot use for\nclosed-source submission.",
         AMBER),
        (bx[2], 1.9,
         "Experiment label: illustrative,\nunvalidated, synthetic-only -\nnot a demonstrated capability.",
         RED_SOFT),
        (bx[3], 1.9,
         "No labeled dataset found after\nlit search. Closed as future work.",
         GREY),
    ]
    for (cx, cy, txt, col) in blockers:
        ax.text(cx, cy, txt, ha="center", va="center",
                fontsize=7.5, color=col, style="italic",
                bbox=dict(facecolor="#111122", edgecolor=col,
                          boxstyle="round,pad=0.3", alpha=0.85),
                zorder=4)
        _arrow(ax, cx, branch_y - 0.55, cx, cy + 0.45, color=col, lw=1.1)

    ax.text(7, 0.35,
            "Investigating and documenting blockers is a feature, not a gap - "
            "it shows the system boundary is defined by evidence, not omission.",
            ha="center", va="center", fontsize=8, color=GREY, style="italic")

    path = OUT / "04_class_expansion_decisions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [4] Saved: {path.name}")


# ===========================================================================
if __name__ == "__main__":
    print("")
    print("=== Generating Part B diagrams ===")
    print(f"  Output dir: {OUT}")
    print("")
    diagram_system_architecture()
    diagram_confidence_score()
    diagram_evaluation_charts()
    diagram_class_expansion()
    print("")
    print("Done.")
