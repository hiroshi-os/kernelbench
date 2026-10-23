# kernelbench

Correctness-first experiments with **RMSNorm** and **fused softmax** in Triton.
Both kernels accumulate in FP32, mask non-power-of-two tails, and expose warp
count. A benchmark compares each configuration against the same expression in
eager PyTorch and `torch.compile`, recording correctness, latency distributions,
hardware/software metadata, and both wins and losses in JSON.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hiroshi-os/kernelbench/blob/main/notebooks/gpu_colab.ipynb)

The [Colab notebook](notebooks/gpu_colab.ipynb) preserves the runtime's CUDA
PyTorch, requires real FP32/FP16 GPU tests with zero accepted skips, runs measured
kernel benchmarks, and downloads JSON/Markdown reports and provenance. Choose a
GPU runtime first. It also runs CPU Gloo DDP training; on a single GPU it records
multi-GPU NCCL as **pending**, never passed.

## Current evidence

This repository was developed on Windows without a supported CUDA device.
**Triton kernels have not been compiled or executed on a GPU yet.** CPU tests
validate the reference expressions, API contracts, statistical reporting, and
the report's separation of CPU validation from GPU measurements. GPU tests skip
explicitly on unsupported machines. There are no measured GPU speedups yet.
The original CPU validation artifact records four reference shapes in
`reports/local-validation.json`;
`reports/validation-environment.json` records package versions. No CUDA kernel
compilation or timing was performed. The same CPU test workflow passed on GitHub.
This is an implemented experiment harness awaiting GPU
validation, not a claim of optimized performance.

The separate distributed experiment has now executed on Windows CPU with two
real Gloo workers: all-reduce SUM, DDP gradient averaging, global mean loss and
SGD parameter updates match a serial full-batch reference. See
[local Gloo evidence](reports/local-distributed-gloo.json) and its
[measured summary](reports/local-distributed-gloo.md). This validates CPU
communication and actual training only. CUDA/NCCL and Triton GPU execution remain
pending; no GPU performance or distributed scaling claim is made.
Local validation on 2026-10-23: **39 passed, 26 skipped** (25 Triton GPU cases
and one two-GPU NCCL case). Strict GPU mode separately failed preflight on this
CPU Windows host, as intended; it cannot turn skipped GPU tests into a pass.

## Install and CPU checks

Python 3.10+; a CPU PyTorch installation is sufficient for reference tests.

```sh
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
pip install -e '.[test]'
pytest -q
kernelbench --cpu-validate --output reports/cpu.json
```

## Run the GPU experiment

Use **Linux with an NVIDIA CUDA GPU**, a compatible driver, and CUDA-enabled
PyTorch. Windows CPU mode works; native Windows Triton execution is outside this
project's supported configuration. An AMD integrated GPU is insufficient. Install
the PyTorch CUDA build appropriate to the runner first, then:

```sh
pip install -e '.[gpu,test]'
python -m pytest tests/test_gpu.py --require-gpu -k 'not bf16' -q -ra
kernelbench --shape 1x128 --shape 128x781 --shape 1024x4096 \
  --dtype float16 --warps 4 8 16 --warmup 25 --samples 50 --inner 20 \
  --seed 17 --output reports/gpu-float16.json
python -m kernelbench.analyze reports/gpu-float16.json
```

Repeat for `float32` and, if supported by your GPU, `bfloat16`. The manual GPU
workflow requires a self-hosted runner labeled `self-hosted, linux, x64, cuda`;
standard GitHub-hosted CPU runners do not supply a GPU. The regular CI workflow
performs CPU checks and uploads a reference-validation report.

## Method and interpretation

* Seeded inputs are created outside the timed region. Every backend must pass
  numerical comparison before timing. FP32 uses rtol 2e-4 / atol 2e-5; FP16 uses
  2e-3 / 2e-3; BF16 uses 2e-2 / 2e-2. GPU tests include irregular widths and extreme
  finite softmax logits. NaN and infinity inputs have no promised semantics.
* Warm-up excludes JIT/Inductor compilation from steady-state measurements.
  CUDA events surround repeated operation calls on the current stream; the end
  event synchronizes before its elapsed time is read. JSON records inner count,
  sample count, p10, median and p90 milliseconds per operation.
* Every backend allocates its output. The benchmark measures operation calls
  including allocation/launch effects visible to the device; it does not measure
  complete application latency or CPU overhead in isolation.
* `speedup = baseline median / candidate median`. Values above 1.05 are labeled
  wins, below .95 losses; the remainder are within 5%. These labels are a simple
  descriptive threshold, **not statistical significance**. Inspect distributions
  and repeat runs before drawing conclusions. No cache flushing or clock locking
  is performed; other workloads and GPU power settings can affect results.
* Warp sweeps expose resource tradeoffs; the script does not silently select and
  report only a winning configuration. Default shapes include launch-bound short
  rows, irregular masked rows, and larger bandwidth-oriented workloads.
* Eager RMSNorm is a decomposed FP32 reference; eager softmax uses PyTorch's
  existing softmax operation with explicit FP32 conversion. `torch.compile` uses
  the same expressions. These are specified baselines, not a claim of superiority
  over every vendor implementation.

## API scope

`rmsnorm(x, weight, eps=1e-6, num_warps=4)` and
`softmax(x, num_warps=4)` require contiguous, nonempty 2-D CUDA tensors with
float16, bfloat16, or float32 dtype, widths <=32768, and no gradients. RMSNorm's
1-D scale must match width, dtype and device. Epsilon must be finite and positive.
Backward passes, ragged/strided tensors, multi-GPU execution, and training kernels
are outside the Triton API. The separate distributed experiment below uses
PyTorch autograd/DDP, not these inference-only kernels. One program handles one row; very wide rows may
suffer register pressure, which is why a width bound is enforced.

## Distributed communication and training experiment

Use two CPU workers with Gloo, or two distinct visible CUDA GPUs with NCCL:

```sh
torchrun --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
  -m kernelbench.distributed --backend gloo --timeout-seconds 60 \
  --output reports/distributed-gloo.json

torchrun --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
  -m kernelbench.distributed --backend nccl --timeout-seconds 60 \
  --output reports/distributed-nccl.json
```

The NCCL command fails before process-group creation when fewer than two GPUs
are visible. Each NCCL process binds its own `LOCAL_RANK` GPU; two processes on a
single Colab device are not treated as multi-GPU evidence. This harness supports
one node. `python -m torch.distributed.run` is equivalent to `torchrun`.

The communication check reduces rank-dependent FP32 tensors and verifies the
expected sum on every worker before and after timing. A separate tiny two-layer
model runs actual forward/backward/SGD steps on disjoint equal-size data shards.
Each step verifies every averaged DDP gradient and updated parameter against an
identically initialized serial model using the full batch and mean MSE, verifies
the globally averaged loss, and requires a nonzero parameter update. Default
seed is 17, learning rate .05, and five steps. Reports distinguish communication
from training; the experiment is about correctness and overhead, not scaling.

JSON and Markdown are written by rank zero only after all ranks pass. Reports
include backend, worker devices, seed, commit/dirty state, PyTorch/CUDA versions,
driver version when `nvidia-smi` is available, errors and timing distributions.
Synchronized host wall times use the slowest rank per sample. All-reduce timing
excludes input reset and the preceding barrier. DDP timing includes forward,
backward and optimizer step, including the first iteration; excludes serial
reference validation and the preceding barrier. Serial execution is a numerical
oracle, not a timed speedup baseline. Warmup/samples, tensor length, batch size,
steps and process-group timeout are configurable; groups are destroyed in
`finally`, and notebook/test launchers bound and clean up subprocess trees.

`pytest -q` runs the real two-process Gloo check when supported and explicitly
skips the NCCL integration check without two GPUs. A backend available in the
build but failing at runtime fails the test; it is not converted to a skip.
For the CPU-only Windows PyTorch build without libuv, tests use torchrun's
FileStore rendezvous with `USE_LIBUV=0` and
`TORCH_DISABLE_SHARE_RDZV_TCP_STORE=1`. Linux/Colab use the standard launcher
commands above. CPU CI archives a fresh Gloo JSON/Markdown report. The manual
single-GPU CI runs strict FP32/FP16 kernel checks; multi-GPU NCCL must be launched
separately on suitable hardware.

## Next experiments

1. Run the checked-in GPU tests and archive the first real GPU report.
2. Compare row-width and warp-count effects across at least two GPUs.
3. Profile slow configurations with Nsight Compute, recording register use,
   occupancy and achieved memory bandwidth before changing the kernel design.

## Sources

The kernels use documented Triton reduction/masking primitives. The softmax
reduction pattern is adapted from the [official fused softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).
See also [Triton sum](https://triton-lang.org/main/python-api/generated/triton.language.sum.html).
See [NOTICE](NOTICE) for attribution and [LICENSE](LICENSE) for licensing.
Distributed APIs follow the official [process-group and collective documentation](https://docs.pytorch.org/docs/stable/distributed.html),
[DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
and [torchrun launcher](https://docs.pytorch.org/docs/stable/elastic/run.html).
