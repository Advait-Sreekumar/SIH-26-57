"""Export best_model_v1_iou0.71.pth to ONNX FP32 and INT8.

Fix for AttributeError: cannot import name 'float4_e2m1fn' from 'ml_dtypes':
  torch.onnx.export with opset_version=16 avoids Float4E2M1 (opset 20+ FP8 only).

Usage:
    python export_onnx.py                        # uses runs/best_model_v1_iou0.71.pth
    python export_onnx.py --ckpt path/to/ckpt   # explicit checkpoint path

Outputs (same directory as checkpoint):
    wreck_unet.onnx        FP32, opset 16
    wreck_unet_int8.onnx   INT8 dynamic quantization (QUInt8)
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import segmentation_models_pytorch as smp
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort


def export(ckpt_path, opset=16):
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)

    # Infer attention type from checkpoint keys; fall back to None if absent.
    has_scse = any("attention" in k for k in ckpt["model_state_dict"])
    model = smp.Unet(
        encoder_name=ckpt.get("encoder", "resnet34"),
        encoder_weights=None,
        in_channels=1,
        classes=1,
        decoder_attention_type="scse" if has_scse else None,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir = ckpt_path.parent
    fp32_path = out_dir / "wreck_unet.onnx"
    int8_path = out_dir / "wreck_unet_int8.onnx"

    rng = np.random.default_rng(42)
    dummy_np = rng.standard_normal((1, 1, 512, 512)).astype(np.float32) * 0.3
    dummy_t = torch.from_numpy(dummy_np)

    # Export FP32
    torch.onnx.export(
        model,
        dummy_t,
        str(fp32_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch", 2: "H", 3: "W"},
            "logits": {0: "batch", 2: "H", 3: "W"},
        },
    )
    print(f"FP32 exported: {fp32_path}")

    # Validate FP32 vs PyTorch
    with torch.no_grad():
        pt_out = model(dummy_t).numpy()

    sess32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    ort_out = sess32.run(None, {"input": dummy_np})[0]
    diff_fp32 = float(np.abs(pt_out - ort_out).max())
    status_fp32 = "PASS" if diff_fp32 < 1e-4 else "WARN"
    print(f"FP32 validation: max|PyTorch-ONNX| = {diff_fp32:.2e}  [{status_fp32}]")

    # Export INT8 via dynamic quantization
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)
    print(f"INT8 exported: {int8_path}")

    # Validate INT8 vs PyTorch
    sess8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    ort8_out = sess8.run(None, {"input": dummy_np})[0]
    diff_int8 = float(np.abs(pt_out - ort8_out).max())
    status_int8 = "PASS" if diff_int8 < 0.5 else "WARN"
    print(f"INT8 validation: max|PyTorch-INT8| = {diff_int8:.4f}  [{status_int8}]")

    return diff_fp32, diff_int8


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export U-Net checkpoint to ONNX")
    _default_ckpt = Path(__file__).parent / "runs" / "best_model_v1_iou0.71.pth"
    ap.add_argument("--ckpt", default=str(_default_ckpt), help="Path to .pth checkpoint")
    ap.add_argument("--opset", type=int, default=16, help="ONNX opset version (default 16)")
    args = ap.parse_args()
    export(args.ckpt, args.opset)
