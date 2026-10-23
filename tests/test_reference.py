import pytest
import torch

from kernelbench.ops import gpu_status, rmsnorm, rmsnorm_reference, softmax_reference, validate


@pytest.mark.parametrize("shape", [(1, 1), (7, 33), (13, 781), (2, 4096)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reference(shape, dtype):
    torch.manual_seed(2)
    x = torch.randn(shape).to(dtype)
    w = torch.randn(shape[-1]).to(dtype)
    actual = rmsnorm_reference(x, w)
    expected = torch.nn.functional.rms_norm(x.float(), (shape[-1],), w.float(), eps=1e-6).to(dtype)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == dtype
    torch.testing.assert_close(softmax_reference(x).float().sum(-1), torch.ones(shape[0]), atol=.01, rtol=.01)


def test_softmax_extremes():
    x = torch.tensor([[10000., -10000., 0.], [-10000., -10000., -10000.]])
    actual = softmax_reference(x)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.sum(-1), torch.ones(2))


@pytest.mark.parametrize("x", [torch.empty(0, 3), torch.ones(3), torch.ones(1, 2, 3),
                                    torch.ones(2, 3, dtype=torch.int64), torch.ones(2, 3).T,
                                    torch.ones(1, 32769), torch.ones(2, 3, requires_grad=True)])
def test_invalid_inputs(x):
    with pytest.raises(ValueError):
        validate(x)


@pytest.mark.parametrize("eps", [0, -1, float("inf"), float("nan")])
def test_invalid_epsilon(eps):
    with pytest.raises(ValueError):
        rmsnorm_reference(torch.ones(2, 3), torch.ones(3), eps)


def test_invalid_weight():
    for w in [torch.ones(4), torch.ones(3, dtype=torch.float16), torch.ones(6)[::2]]:
        with pytest.raises(ValueError):
            validate(torch.ones(2, 3), w)


def test_unsupported_runtime_is_explicit():
    if gpu_status() is None:
        pytest.skip("Supported GPU runtime present")
    with pytest.raises(RuntimeError, match="Linux|CUDA|gpu extra"):
        rmsnorm(torch.ones(2, 3), torch.ones(3))


def test_reference_expressions_trace_as_full_graph():
    # Validate Dynamo tracing without implying CUDA/Inductor kernel execution.
    x, w = torch.randn(3, 33), torch.randn(33)
    for fn in (lambda: rmsnorm_reference(x, w), lambda: softmax_reference(x)):
        traced = torch.compile(fn, backend="eager", fullgraph=True)
        torch.testing.assert_close(traced(), fn())
