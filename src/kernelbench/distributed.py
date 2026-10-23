"""Single-node torchrun experiment: real collective and DDP/SGD correctness.

Gloo uses CPU; NCCL uses one distinct CUDA device per process. These experiments
use PyTorch training, not the inference-only Triton kernels in this repository.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import socket
from time import perf_counter

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .benchmark import positive, summarize
from .environment import provenance


def preflight(backend, world_size, local_world_size, local_rank):
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    if world_size < 2 or world_size != local_world_size:
        raise RuntimeError("Use a single-node torchrun launcher with at least two workers")
    if not 0 <= local_rank < local_world_size:
        raise RuntimeError("Invalid LOCAL_RANK from launcher")
    if backend == "nccl":
        if torch.cuda.device_count() < 2:
            raise RuntimeError("NCCL requires at least two visible CUDA GPUs; single-GPU validation is pending")
        if not dist.is_nccl_available() or not torch.cuda.is_available():
            raise RuntimeError("NCCL and CUDA-enabled PyTorch are required")
        if local_world_size > torch.cuda.device_count():
            raise RuntimeError("NCCL requires one distinct CUDA GPU per worker")
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    if not dist.is_gloo_available():
        raise RuntimeError("This PyTorch build does not support Gloo")
    return torch.device("cpu")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def communication(device, rank, world_size, elements, warmup, samples):
    base = torch.linspace(-1, 1, elements, device=device)
    expected = base * (world_size * (world_size + 1) / 2)
    value = base * (rank + 1)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    torch.testing.assert_close(value, expected, rtol=1e-5, atol=1e-6)
    max_error = (value - expected).abs().max().item()
    times = []
    for iteration in range(warmup + samples):
        value.copy_(base * (rank + 1))
        dist.barrier()
        synchronize(device)
        start = perf_counter()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        synchronize(device)
        elapsed = (perf_counter() - start) * 1000
        if iteration >= warmup:
            times.append(elapsed)
    torch.testing.assert_close(value, expected, rtol=1e-5, atol=1e-6)
    return {"correctness": "passed", "max_abs_error": max_error,
            "elements": elements, "bytes_per_rank": elements * value.element_size(),
            "dtype": str(value.dtype), "samples_ms": times}


def training(device, rank, world_size, seed, steps, batch_per_rank):
    # Identical initialization and global data; each worker uses a disjoint,
    # equal-sized shard. DDP's averaged gradient matches global mean MSE.
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)).to(device)
    serial = deepcopy(model)
    initial = [p.detach().clone() for p in model.parameters()]
    ddp = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.05)
    serial_optimizer = torch.optim.SGD(serial.parameters(), lr=0.05)
    generator = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(world_size * batch_per_rank, 8, generator=generator).to(device)
    y = torch.randn(world_size * batch_per_rank, 2, generator=generator).to(device)
    shard = slice(rank * batch_per_rank, (rank + 1) * batch_per_rank)
    gradient_error, parameter_error = 0., 0.
    times, losses = [], []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        serial_optimizer.zero_grad(set_to_none=True)
        serial_loss = torch.nn.functional.mse_loss(serial(x), y)
        serial_loss.backward()
        dist.barrier()
        synchronize(device)
        start = perf_counter()
        loss = torch.nn.functional.mse_loss(ddp(x[shard]), y[shard])
        loss.backward()
        optimizer.step()
        synchronize(device)
        times.append((perf_counter() - start) * 1000)
        for distributed_parameter, serial_parameter in zip(model.parameters(), serial.parameters()):
            torch.testing.assert_close(distributed_parameter.grad, serial_parameter.grad, rtol=2e-5, atol=2e-6)
            gradient_error = max(gradient_error, (distributed_parameter.grad - serial_parameter.grad).abs().max().item())
        serial_optimizer.step()
        for distributed_parameter, serial_parameter in zip(model.parameters(), serial.parameters()):
            torch.testing.assert_close(distributed_parameter, serial_parameter, rtol=2e-5, atol=2e-6)
            parameter_error = max(parameter_error, (distributed_parameter - serial_parameter).abs().max().item())
        # This checks collective loss agreement separately from gradient checks.
        global_loss = loss.detach().clone()
        dist.all_reduce(global_loss)
        global_loss /= world_size
        torch.testing.assert_close(global_loss, serial_loss.detach(), rtol=2e-5, atol=2e-6)
        losses.append(global_loss.item())
    delta = max((p.detach() - original).abs().max().item() for p, original in zip(model.parameters(), initial))
    if delta <= 0:
        raise AssertionError("Training must actually update model parameters")
    return {"gradient_equivalence": "passed", "parameter_equivalence": "passed",
            "global_mean_loss_equivalence": "passed", "optimizer": "SGD", "learning_rate": 0.05,
            "steps": steps, "batch_per_rank": batch_per_rank, "dtype": "torch.float32",
            "max_gradient_abs_error": gradient_error, "max_parameter_abs_error": parameter_error,
            "parameter_update_max_abs": delta, "global_mean_losses": losses, "samples_ms": times}


def markdown(report):
    comm, train = report["communication"], report["training"]
    return (f"# Distributed experiment: {report['backend']} / {report['device_type']}\n\n"
            f"Workers: {report['world_size']}; seed: {report['seed']}; commit: {report['git_commit']}\n\n"
            "| Experiment | Correctness | Slowest-rank median ms |\n|---|---|---:|\n"
            f"| All-reduce SUM ({comm['elements']} FP32 elements) | {comm['correctness']} | {comm['latency']['median_ms']:.6f} |\n"
            f"| DDP forward/backward/SGD | gradients + parameters + global mean loss passed | {train['latency']['median_ms']:.6f} |\n\n"
            f"Maximum gradient error: {train['max_gradient_abs_error']:.8g}; parameter error: {train['max_parameter_abs_error']:.8g}.\n\n"
            "Times are synchronized host wall time; each sample uses the slowest worker. "
            "Communication excludes input reset and the preceding barrier. Training includes DDP forward, "
            "backward and SGD, excludes serial checks and the preceding barrier, and includes the first step. "
            "No speedup or scaling claim is made. CPU Gloo does not validate CUDA NCCL.\n")


def run(args):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    device = preflight(args.backend, world, local_world, local_rank)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        dist.init_process_group(args.backend, timeout=timedelta(seconds=args.timeout_seconds))
        comm = communication(device, rank, world, args.elements, args.warmup, args.samples)
        train = training(device, rank, world, args.seed, args.steps, args.batch_per_rank)
        local = {"rank": rank, "device": str(device), "hostname": socket.gethostname(),
                 "communication": comm, "training": train}
        workers = [None] * world
        dist.all_gather_object(workers, local)
        if rank == 0:
            for key, result in (("communication", comm), ("training", train)):
                # Per-sample maximum across workers avoids optimistic rank-0-only timing.
                result["latency"] = summarize([max(samples) for samples in zip(
                    *(worker[key]["samples_ms"] for worker in workers))])
            comm["max_abs_error"] = max(w["communication"]["max_abs_error"] for w in workers)
            for key in ("max_gradient_abs_error", "max_parameter_abs_error"):
                train[key] = max(w["training"][key] for w in workers)
            report = {"schema_version": 1, "status": "distributed_experiment_passed",
                      "created_at": datetime.now(timezone.utc).isoformat(), **provenance(),
                      "backend": args.backend, "device_type": device.type, "world_size": world,
                      "seed": args.seed, "timeout_seconds": args.timeout_seconds,
                      "communication_warmup": args.warmup,
                      "timing_method": "synchronized_host_wall_time_slowest_rank_per_sample",
                      "communication": comm, "training": train, "workers": workers}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
            print(f"{report['status']}: {args.backend} {device.type}, {world} workers; {args.output}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gloo", "nccl"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--elements", type=positive, default=65536)
    parser.add_argument("--warmup", type=positive, default=3)
    parser.add_argument("--samples", type=positive, default=10)
    parser.add_argument("--steps", type=positive, default=5)
    parser.add_argument("--batch-per-rank", type=positive, default=8)
    parser.add_argument("--timeout-seconds", type=positive, default=60)
    args = parser.parse_args()
    try:
        run(args)
    except RuntimeError as exc:
        parser.exit(2, f"Distributed experiment not completed: {exc}\n")


if __name__ == "__main__":
    main()
