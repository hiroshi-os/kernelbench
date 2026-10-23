import json
import os
from pathlib import Path
import sys

import pytest
import torch
import torch.distributed as dist

from kernelbench.distributed import preflight
from kernelbench.launch import bounded_run


def test_nccl_rejects_single_gpu_before_process_group(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="at least two visible CUDA GPUs"):
        preflight("nccl", 2, 2, 0)


def run_experiment(tmp_path, backend):
    output = tmp_path / f"{backend}.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src") + os.pathsep + env.get("PYTHONPATH", "")
    # PyTorch's Windows build can lack libuv; this selects the documented TCPStore fallback.
    env["USE_LIBUV"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc-per-node=2",
               "--rdzv-backend=c10d", "--rdzv-endpoint=127.0.0.1:0", "--max-restarts=0",
               "-m", "kernelbench.distributed", "--backend", backend, "--output", str(output),
               "--elements", "1024", "--warmup", "1", "--samples", "2", "--steps", "3",
               "--timeout-seconds", "30"]
    if os.name == "nt":
        # Windows torchrun's TCP rendezvous constructs a libuv store directly.
        # File rendezvous plus the non-shared bootstrap uses USE_LIBUV=0.
        command[command.index("--rdzv-endpoint=127.0.0.1:0")] = f"--rdzv-endpoint={tmp_path / 'rendezvous'}"
        command.insert(6, "--rdzv-conf=store_type=file")
        env["TORCH_DISABLE_SHARE_RDZV_TCP_STORE"] = "1"
    result = bounded_run(command, timeout=120, env=env)
    assert "distributed_experiment_passed" in result.stdout
    report = json.loads(output.read_text())
    assert report["backend"] == backend
    assert report["world_size"] == 2
    assert sorted(worker["rank"] for worker in report["workers"]) == [0, 1]
    assert report["communication"]["correctness"] == "passed"
    assert report["communication"]["latency"]["samples"] == 2
    assert report["communication"]["latency"]["median_ms"] > 0
    training = report["training"]
    assert training["steps"] == 3
    assert training["gradient_equivalence"] == "passed"
    assert training["parameter_equivalence"] == "passed"
    assert training["global_mean_loss_equivalence"] == "passed"
    assert training["parameter_update_max_abs"] > 0
    assert training["latency"]["samples"] == 3
    assert training["latency"]["median_ms"] > 0
    assert "DDP forward/backward/SGD" in output.with_suffix(".md").read_text()
    return report


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="CPU Gloo backend unavailable")
def test_two_process_gloo_communication_and_actual_ddp_training(tmp_path):
    report = run_experiment(tmp_path, "gloo")
    assert report["device_type"] == "cpu"
    assert all(worker["device"] == "cpu" for worker in report["workers"])


@pytest.mark.skipif(not dist.is_available() or not dist.is_nccl_available() or torch.cuda.device_count() < 2,
                    reason="Actual NCCL training requires at least two visible CUDA GPUs and NCCL")
def test_two_gpu_nccl_communication_and_actual_ddp_training(tmp_path):
    report = run_experiment(tmp_path, "nccl")
    assert report["device_type"] == "cuda"
    assert {worker["device"] for worker in report["workers"]} == {"cuda:0", "cuda:1"}
