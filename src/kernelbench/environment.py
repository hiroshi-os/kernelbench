"""Best-effort provenance; missing tools are reported as unavailable."""

from pathlib import Path
import os
import platform
import subprocess

import torch


def command_output(command, cwd=None):
    try:
        return subprocess.run(command, cwd=cwd, check=True, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def provenance():
    root = Path(__file__).resolve().parents[2]
    commit = command_output(["git", "rev-parse", "HEAD"], root)
    dirty = command_output(["git", "status", "--porcelain"], root)
    return {"git_commit": commit, "git_dirty": bool(dirty) if commit else None,
            "python": platform.python_version(), "platform": platform.platform(),
            "cpu_processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER"),
            "logical_cpu_count": os.cpu_count(),
            "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "cuda_driver_nvidia_smi": command_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
            "cuda_device_count": torch.cuda.device_count(),
            "devices": [{"index": i, "name": torch.cuda.get_device_name(i),
                         "capability": list(torch.cuda.get_device_capability(i)),
                         "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                        for i in range(torch.cuda.device_count())]}
