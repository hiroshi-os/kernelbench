import pytest

from kernelbench.analyze import markdown


def test_refuses_to_turn_cpu_validation_into_gpu_results():
    with pytest.raises(ValueError, match="measured GPU"):
        markdown({"status": "cpu_reference_validation_only"})


def test_renders_measured_losses_and_wins():
    # Unit fixture only: not a benchmark artifact or performance claim.
    report = {"status": "gpu_benchmark", "device": "unit-test-device", "dtype": "float32",
              "results": [{"operation": "rmsnorm", "shape": [2, 3], "backends": {
                  "triton_warps_4": {"median_ms": 2.,
                      "versus_eager": {"speedup": .5, "outcome": "loss"},
                      "versus_compile": {"speedup": 2., "outcome": "win"}}}}]}
    rendered = markdown(report)
    assert "0.500x (loss)" in rendered
    assert "2.000x (win)" in rendered
    assert "2x3" in rendered
