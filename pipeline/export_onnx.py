import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from infer import load_detector

MODEL_DIR = Path(r"D:\SIH\AI4Shipwrecks\AI4Shipwrecks\runs")


def export_onnx(out_path=MODEL_DIR / "wreck_unet.onnx", size=512):
    model = load_detector()
    model = model.cpu().eval()
    dummy = torch.randn(1, 1, size, size)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch", 2: "h", 3: "w"}, "logits": {0: "batch", 2: "h", 3: "w"}},
        opset_version=17,
    )
    return out_path


def benchmark(model_path, size=512, warmup=5, iters=30, providers=None):
    sess = ort.InferenceSession(str(model_path), providers=providers or ort.get_available_providers())
    x = np.random.rand(1, 1, size, size).astype(np.float32)
    inp = {s.name: x for s in sess.get_inputs()}
    for _ in range(warmup):
        sess.run(None, inp)
    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(None, inp)
    dt = (time.perf_counter() - t0) / iters
    return dt * 1000.0


def quantize_int8(fp32_path, out_path=MODEL_DIR / "wreck_unet_int8.onnx"):
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(str(fp32_path), str(out_path), weight_type=QuantType.QUInt8)
    return out_path


@torch.no_grad()
def compare_outputs(torch_model, onnx_path, size=512):
    import onnxruntime as ort

    x = np.random.rand(1, 1, size, size).astype(np.float32)
    sess = ort.InferenceSession(str(onnx_path), providers=ort.get_available_providers())
    out_onnx = sess.run(None, {"image": x})[0]
    t = torch.from_numpy(x)
    out_torch = torch_model(t).detach().numpy()
    diff = np.abs(out_onnx - out_torch).mean()
    return diff


def main():
    print("Exporting FP32 ONNX ...")
    onnx_path = export_onnx()
    print(f"  saved: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    print("Quantizing to INT8 ...")
    int8_path = quantize_int8(onnx_path)
    print(f"  saved: {int8_path} ({int8_path.stat().st_size / 1e6:.1f} MB)")

    print("\nBenchmarks (512x512, CPU):")
    ms_fp32 = benchmark(onnx_path)
    ms_int8 = benchmark(int8_path)
    torch_model = load_detector(device="cpu")
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            torch_model(torch.rand(1, 1, 512, 512))
    ms_torch = (time.perf_counter() - t0) / 10 * 1000

    print(f"  PyTorch CPU : {ms_torch:7.1f} ms")
    print(f"  ONNX FP32   : {ms_fp32:7.1f} ms")
    print(f"  ONNX INT8   : {ms_int8:7.1f} ms")

    diff = compare_outputs(torch_model, onnx_path)
    print(f"\nFP32 ONNX vs PyTorch mean abs logit diff: {diff:.5f}")


if __name__ == "__main__":
    main()
