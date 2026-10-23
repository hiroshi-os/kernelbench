"""Row reductions, FP32 accumulation, masked tail lanes, fused scale/store.

Softmax reduction idiom follows Triton's documented tutorial (see NOTICE).
These kernels intentionally expose warp count for reproducible experiments.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm(X, W, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(X + row * N + col, col < N, other=0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(value * value, axis=0) / N + EPS)
    weight = tl.load(W + col, col < N, other=0).to(tl.float32)
    tl.store(Y + row * N + col, value * scale * weight, col < N)


@triton.jit
def _softmax(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(X + row * N + col, col < N, other=-float("inf")).to(tl.float32)
    numerator = tl.exp(value - tl.max(value, axis=0))
    result = numerator / tl.sum(numerator, axis=0)
    tl.store(Y + row * N + col, result, col < N)


def launch_rmsnorm(x, weight, eps, num_warps):
    output = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _rmsnorm[(x.shape[0],)](x, weight, output, x.shape[1], eps,
                               triton.next_power_of_2(x.shape[1]), num_warps=num_warps)
    return output


def launch_softmax(x, num_warps):
    output = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _softmax[(x.shape[0],)](x, output, x.shape[1],
                               triton.next_power_of_2(x.shape[1]), num_warps=num_warps)
    return output
