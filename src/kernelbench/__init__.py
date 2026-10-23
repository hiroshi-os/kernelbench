"""Inference-only kernels with lazy Triton imports."""

from .ops import rmsnorm, softmax

__all__ = ["rmsnorm", "softmax"]
