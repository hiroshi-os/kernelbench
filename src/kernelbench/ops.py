"""Public operations; validate contracts before importing a GPU runtime."""

import math
import platform

import torch


def validate(x, weight=None, eps=1e-6):
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError("expected a nonempty [rows, columns] tensor")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported dtypes: float16, bfloat16, float32")
    if not x.is_contiguous():
        raise ValueError("input must be contiguous")
    if x.shape[1] > 32768:
        raise ValueError("row width must be <=32768; this kernel uses one program per row")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if x.requires_grad or (weight is not None and weight.requires_grad):
        raise ValueError("inference-only API; detach tensors before benchmarking")
    if weight is not None:
        if weight.shape != (x.shape[1],) or not weight.is_contiguous():
            raise ValueError("weight must be contiguous with shape [columns]")
        if weight.device != x.device or weight.dtype != x.dtype:
            raise ValueError("weight must match input device and dtype")


def gpu_status():
    if platform.system() != "Linux":
        return "Triton GPU execution requires Linux in this repository (WSL2 with a supported GPU also works)"
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return "A supported NVIDIA CUDA GPU and CUDA-enabled PyTorch are required"
    try:
        import triton  # noqa: F401
    except ImportError:
        return "Install the gpu extra: pip install -e '.[gpu,test]'"
    return None


def _runtime(x):
    reason = gpu_status()
    if reason:
        raise RuntimeError(reason)
    if x.device.type != "cuda":
        raise ValueError("Triton operations require CUDA tensors")
    from . import kernels
    return kernels


def rmsnorm_reference(x, weight, eps=1e-6):
    validate(x, weight, eps)
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def softmax_reference(x):
    validate(x)
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def rmsnorm(x, weight, eps=1e-6, num_warps=4):
    validate(x, weight, eps)
    if num_warps not in (4, 8, 16):
        raise ValueError("num_warps must be 4, 8, or 16")
    return _runtime(x).launch_rmsnorm(x, weight, eps, num_warps)


def softmax(x, num_warps=4):
    validate(x)
    if num_warps not in (4, 8, 16):
        raise ValueError("num_warps must be 4, 8, or 16")
    return _runtime(x).launch_softmax(x, num_warps)
