# kernelbench

Correctness-first experiments with **RMSNorm** and **fused softmax** in Triton.
Both kernels accumulate in FP32, mask non-power-of-two tails, and expose warp
count. A benchmark compares each configuration against the same expression in
eager PyTorch and `torch.compile`, recording correctness, latency distributions,
hardware/software metadata, and both wins and losses in JSON.

## Current evidence

This repository was developed on Windows without a supported CUDA device.
**Triton kernels have not been compiled or executed on a GPU yet.** CPU tests
validate the reference expressions, API contracts, statistical reporting, and
the report's separation of CPU validation from GPU measurements. GPU tests skip
explicitly on unsupported machines. There are no measured GPU speedups yet.
See [reports/local-validation.json](reports/local-validation.json) for actual
local reference checks. This is an implemented experiment harness awaiting GPU
validation, not a claim of optimized performance.

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
pytest -q
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
are outside this experiment. One program handles one row; very wide rows may
suffer register pressure, which is why a width bound is enforced.

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
