"""Measure inference speed (FPS / latency) for a YOLO model.

Usage::

    python scripts/benchmark_speed.py --weights runs/fsod_baseline/novel_finetune/weights/best.pt
    python scripts/benchmark_speed.py --weights yolo11s.pt --imgsz 640 --device 0

Reports:
  - Warm-up + N iterations
  - Mean / Std latency (ms)
  - FPS
  - Model parameter count
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO inference speed.")
    parser.add_argument("--weights", type=str, required=True, help="Model checkpoint (.pt).")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--device", type=str, default="0", help="Device (0 for GPU, cpu for CPU).")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=300, help="Benchmark iterations.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size.")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference.")
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    args = parse_args()
    weights_path = resolve_repo_path(args.weights)

    device_str = f"cuda:{args.device}" if args.device.isdigit() else args.device
    device = torch.device(device_str)

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    # Access underlying torch model
    torch_model = model.model
    torch_model.to(device)
    torch_model.eval()
    if args.half and device.type == "cuda":
        torch_model.half()

    n_params = count_parameters(torch_model)
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    dtype = torch.float16 if (args.half and device.type == "cuda") else torch.float32
    dummy_input = torch.randn(args.batch, 3, args.imgsz, args.imgsz, device=device, dtype=dtype)

    # Warmup
    print(f"Warming up ({args.warmup} iters)...")
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = torch_model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({args.iters} iters, batch={args.batch})...")
    latencies = []
    with torch.no_grad():
        for _ in range(args.iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = torch_model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

    latencies_tensor = torch.tensor(latencies)
    mean_ms = latencies_tensor.mean().item()
    std_ms = latencies_tensor.std().item()
    fps = 1000.0 / mean_ms * args.batch

    print("=" * 50)
    print(f"Model       : {weights_path.name}")
    print(f"Parameters  : {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"Input       : {args.batch}x3x{args.imgsz}x{args.imgsz}")
    print(f"Device      : {device}")
    print(f"FP16        : {args.half}")
    print(f"Latency     : {mean_ms:.2f} ± {std_ms:.2f} ms")
    print(f"FPS         : {fps:.1f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
