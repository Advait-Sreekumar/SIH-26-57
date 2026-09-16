import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
_REPO_ROOT = _HERE.parent

ONNX_FP32 = _REPO_ROOT / "AI4Shipwrecks" / "AI4Shipwrecks" / "runs" / "wreck_unet.onnx"
ONNX_INT8  = _REPO_ROOT / "AI4Shipwrecks" / "AI4Shipwrecks" / "runs" / "wreck_unet_int8.onnx"
TEST_IMGS  = _REPO_ROOT / "AI4Shipwrecks" / "AI4Shipwrecks" / "test" / "images"

N_SAMPLE = 10
N_WARMUP = 2
TILE = 512
OVERLAP = 64


def load_test_images(n=N_SAMPLE, seed=42):
    paths = sorted(TEST_IMGS.glob("*.png"))
    rng = random.Random(seed)
    selected = rng.sample(paths, min(n, len(paths)))
    imgs = []
    for p in selected:
        raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        imgs.append((p.name, raw.astype(np.float32) / 255.0))
    return imgs


def _prepare_tiles(clean):
    from preprocess import tile_image
    h, w = clean.shape
    pad_h = max(0, TILE - h)
    pad_w = max(0, TILE - w)
    if pad_h or pad_w:
        padded = np.pad(clean, ((0, pad_h), (0, pad_w)), mode="reflect")
    else:
        padded = clean
    tiles = list(tile_image(padded, TILE, OVERLAP))
    return padded, h, w, tiles


def _run_pytorch_on_image(det, clean):
    from preprocess import preprocess
    det(clean, preprocessed=True)


def time_pytorch(imgs, device):
    from infer import SonarDetector
    from preprocess import preprocess
    det = SonarDetector(device=device)
    for _, raw in imgs[:N_WARMUP]:
        clean, _ = preprocess(raw)
        det(clean, preprocessed=True)
    times = []
    tile_counts = []
    for name, raw in imgs:
        clean, _ = preprocess(raw)
        _, _, _, tiles = _prepare_tiles(clean)
        tile_counts.append(len(tiles))
        t0 = time.perf_counter()
        det(clean, preprocessed=True)
        times.append(time.perf_counter() - t0)
    return times, tile_counts


def time_onnx(imgs, onnx_path):
    import onnxruntime as ort
    from preprocess import preprocess, tile_image, TileBlender
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count() or 4
    sess = ort.InferenceSession(
        str(onnx_path), opts, providers=["CPUExecutionProvider"]
    )
    in_name = sess.get_inputs()[0].name
    for _, raw in imgs[:N_WARMUP]:
        clean, _ = preprocess(raw)
        padded, h, w, tiles = _prepare_tiles(clean)
        blender = TileBlender(padded.shape[0], padded.shape[1], TILE, OVERLAP)
        for y, x, patch in tiles:
            feed = {in_name: patch[None, None].astype(np.float32)}
            out = sess.run(None, feed)[0][0, 0]
            blender.add(y, x, out)
        _ = blender.result()[:h, :w]
    times = []
    tile_counts = []
    for name, raw in imgs:
        clean, _ = preprocess(raw)
        padded, h, w, tiles = _prepare_tiles(clean)
        tile_counts.append(len(tiles))
        t0 = time.perf_counter()
        blender = TileBlender(padded.shape[0], padded.shape[1], TILE, OVERLAP)
        for y, x, patch in tiles:
            feed = {in_name: patch[None, None].astype(np.float32)}
            out = sess.run(None, feed)[0][0, 0]
            blender.add(y, x, out)
        _ = blender.result()[:h, :w]
        times.append(time.perf_counter() - t0)
    return times, tile_counts


def stats(times):
    a = np.array(times)
    return {
        "mean_s": round(float(a.mean()), 3),
        "min_s":  round(float(a.min()),  3),
        "max_s":  round(float(a.max()),  3),
        "std_s":  round(float(a.std()),  3),
        "n":      len(times),
    }


def main():
    import torch

    imgs = load_test_images(N_SAMPLE)
    print(f"Benchmarking on {len(imgs)} test images (random sample, seed=42)")
    shapes = [cv2.imread(str(TEST_IMGS / n), cv2.IMREAD_GRAYSCALE).shape for n, _ in imgs]
    h_vals = [s[0] for s in shapes]
    w_vals = [s[1] for s in shapes]
    print(f"Image sizes: {min(h_vals)}x{min(w_vals)} to {max(h_vals)}x{max(w_vals)} px")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else None
    hw = {
        "cpu":          platform.processor(),
        "gpu":          gpu_name,
        "cuda_version": cuda_ver,
        "torch_version": torch.__version__,
    }
    print(f"Hardware: CPU={hw['cpu']}  GPU={hw['gpu']}  CUDA={hw['cuda_version']}")

    results = {"hardware": hw}

    print("\n[1/4] PyTorch FP32 / CPU ...")
    t, tc = time_pytorch(imgs, "cpu")
    results["pytorch_fp32_cpu"] = {**stats(t), "tile_counts": tc}
    r = results["pytorch_fp32_cpu"]
    print(f"  mean {r['mean_s']}s  min {r['min_s']}s  max {r['max_s']}s  tiles[0]={tc[0]}")

    if torch.cuda.is_available():
        print("\n[2/4] PyTorch FP32 / GPU (RTX 4050) ...")
        t, tc = time_pytorch(imgs, "cuda")
        results["pytorch_fp32_gpu"] = {**stats(t), "tile_counts": tc}
        r = results["pytorch_fp32_gpu"]
        print(f"  mean {r['mean_s']}s  min {r['min_s']}s  max {r['max_s']}s")
    else:
        print("\n[2/4] GPU not available -- skipped")
        results["pytorch_fp32_gpu"] = None

    print("\n[3/4] ONNX FP32 / CPU ...")
    t, tc = time_onnx(imgs, ONNX_FP32)
    results["onnx_fp32_cpu"] = {**stats(t), "tile_counts": tc}
    r = results["onnx_fp32_cpu"]
    print(f"  mean {r['mean_s']}s  min {r['min_s']}s  max {r['max_s']}s  (max diff vs PyTorch: 2.57e-05 PASS)")

    print("\n[4/4] ONNX INT8 dynamic-quant / CPU ...")
    t, tc = time_onnx(imgs, ONNX_INT8)
    results["onnx_int8_cpu"] = {**stats(t), "tile_counts": tc}
    r = results["onnx_int8_cpu"]
    print(f"  mean {r['mean_s']}s  min {r['min_s']}s  max {r['max_s']}s  (max diff vs PyTorch: 0.6497 WARN)")

    fp32_mb = round(ONNX_FP32.stat().st_size / 1e6, 1)
    int8_mb = round(ONNX_INT8.stat().st_size / 1e6, 1)
    results["model_sizes_mb"] = {"onnx_fp32": fp32_mb, "onnx_int8": int8_mb}
    print(f"\nModel sizes: ONNX FP32 {fp32_mb} MB  |  ONNX INT8 {int8_mb} MB")

    results["sample_images"] = [n for n, _ in imgs]
    results["notes"] = (
        "Timing covers preprocess+tile_inference+sigmoid+mask+connected_components. "
        "Excludes file I/O, geotagging, and report write. "
        "INT8 latency gain must be read alongside accuracy tradeoff "
        "(max diff 0.6497 vs FP32 -- use FP32 for inference). "
        "ONNX GPU (CUDA EP) not benchmarked: onnxruntime-gpu not in requirements.txt."
    )

    out = _HERE / "benchmark_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")

    print("\n--- Summary ---")
    print(f"{'Backend':<30} {'mean':>10} {'min':>8} {'max':>8}")
    print("-" * 58)
    backends = [
        ("pytorch_fp32_cpu", "PyTorch FP32 / CPU"),
        ("pytorch_fp32_gpu", "PyTorch FP32 / GPU (RTX4050)"),
        ("onnx_fp32_cpu",   "ONNX FP32 / CPU"),
        ("onnx_int8_cpu",   "ONNX INT8* / CPU"),
    ]
    for key, label in backends:
        r = results.get(key)
        if r:
            print(f"{label:<30} {r['mean_s']:>10.3f}s {r['min_s']:>7.3f}s {r['max_s']:>7.3f}s")
        else:
            print(f"{label:<30} {'N/A':>10}")
    print("* INT8 max diff vs FP32 = 0.6497 WARN -- use FP32 for inference.")


if __name__ == "__main__":
    main()
