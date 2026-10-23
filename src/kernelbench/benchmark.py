"""CUDA-event benchmark and explicit CPU-only validation report."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch

from .ops import gpu_status, rmsnorm, rmsnorm_reference, softmax, softmax_reference
from .environment import provenance


def quantile(values, fraction):
    if not values or not 0 <= fraction <= 1:
        raise ValueError("nonempty samples and quantile in [0, 1] required")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("timings must be finite and nonnegative")
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def summarize(samples):
    return {"samples": len(samples), "p10_ms": quantile(samples, .1),
            "median_ms": quantile(samples, .5), "p90_ms": quantile(samples, .9)}


def comparison(baseline, candidate):
    if not math.isfinite(baseline) or not math.isfinite(candidate) or baseline <= 0 or candidate <= 0:
        raise ValueError("positive timings required")
    speedup = baseline / candidate
    return {"speedup": speedup, "outcome": "win" if speedup > 1.05 else "loss" if speedup < .95 else "within_5_percent"}


def event_time(fn, warmup, samples, inner):
    # Warm-up includes compilation. Allocation/launch are intentionally part of
    # each operation, while input construction and correctness checks are not.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / inner)
    return summarize(timings)


def metadata(seed):
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            **provenance(), "seed": seed}


def run_cpu_validation(seed=17):
    torch.manual_seed(seed)
    report = metadata(seed)
    report.update(status="cpu_reference_validation_only", gpu_status="not_executed",
                  gpu_reason=gpu_status() or "CPU validation mode explicitly selected", results=[])
    for rows, cols in ((1, 1), (7, 33), (13, 781), (2, 4096)):
        x = torch.randn(rows, cols)
        weight = torch.randn(cols)
        rms = rmsnorm_reference(x, weight)
        expected = torch.nn.functional.rms_norm(x, (cols,), weight, eps=1e-6)
        torch.testing.assert_close(rms, expected)
        sm = softmax_reference(x)
        torch.testing.assert_close(sm.sum(-1), torch.ones(rows))
        report["results"].append({"shape": [rows, cols], "reference_checks": "passed"})
    return report


def run_gpu(shapes, dtype, warps, warmup, samples, inner, seed=17):
    reason = gpu_status()
    if reason:
        raise RuntimeError(reason)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected device does not support bfloat16")
    import triton
    torch.manual_seed(seed)
    report = metadata(seed)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    report.update(status="gpu_benchmark", triton=triton.__version__, device=props.name,
                  capability=list(torch.cuda.get_device_capability()), memory_bytes=props.total_memory,
                  warmup=warmup, samples=samples, inner=inner, dtype=str(dtype), results=[])
    tolerance = {torch.float32: (2e-4, 2e-5), torch.float16: (2e-3, 2e-3),
                 torch.bfloat16: (2e-2, 2e-2)}[dtype]
    for rows, cols in shapes:
        x = torch.randn(rows, cols, device="cuda", dtype=dtype)
        w = torch.randn(cols, device="cuda", dtype=dtype)
        for operation in ("rmsnorm", "softmax"):
            # torch.compile receives the same numerical expression as eager.
            eager = (lambda: rmsnorm_reference(x, w)) if operation == "rmsnorm" else (lambda: softmax_reference(x))
            compiled = torch.compile(eager, fullgraph=True)
            expected = eager()
            row = {"operation": operation, "shape": [rows, cols], "backends": {}}
            for name, fn in (("torch_eager", eager), ("torch_compile", compiled)):
                torch.testing.assert_close(fn(), expected, rtol=tolerance[0], atol=tolerance[1])
                row["backends"][name] = event_time(fn, warmup, samples, inner)
            for num_warps in warps:
                fn = (lambda n=num_warps: rmsnorm(x, w, num_warps=n)) if operation == "rmsnorm" else (lambda n=num_warps: softmax(x, num_warps=n))
                actual = fn()
                torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])
                result = event_time(fn, warmup, samples, inner)
                result["max_abs_error"] = (actual.float() - expected.float()).abs().max().item()
                result["versus_eager"] = comparison(row["backends"]["torch_eager"]["median_ms"], result["median_ms"])
                result["versus_compile"] = comparison(row["backends"]["torch_compile"]["median_ms"], result["median_ms"])
                row["backends"][f"triton_warps_{num_warps}"] = result
            report["results"].append(row)
    return report


def parse_shape(value):
    try:
        rows, cols = map(int, value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be ROWSxCOLS") from exc
    if rows <= 0 or cols <= 0 or cols > 32768:
        raise argparse.ArgumentTypeError("positive shape required; columns <=32768")
    return rows, cols


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-validate", action="store_true")
    parser.add_argument("--shape", action="append", type=parse_shape)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--warps", nargs="+", choices=(4, 8, 16), type=int, default=[4, 8])
    parser.add_argument("--warmup", type=positive, default=25)
    parser.add_argument("--samples", type=positive, default=50)
    parser.add_argument("--inner", type=positive, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_cpu_validation(args.seed) if args.cpu_validate else run_gpu(
            args.shape or [(1, 128), (128, 781), (1024, 4096)], getattr(torch, args.dtype),
            args.warps, args.warmup, args.samples, args.inner, args.seed)
    except RuntimeError as exc:
        parser.exit(2, f"GPU benchmark not completed: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{report['status']}: {args.output}")


if __name__ == "__main__":
    main()
