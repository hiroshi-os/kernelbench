import pytest
import torch

from kernelbench.ops import gpu_status, rmsnorm, rmsnorm_reference, softmax, softmax_reference

pytestmark = pytest.mark.skipif(gpu_status() is not None, reason=gpu_status() or "GPU available")


@pytest.mark.parametrize("shape", [(1, 1), (7, 33), (13, 781), (64, 4096)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("warps", [4, 8])
def test_kernels(shape, dtype, warps):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("device does not support BF16")
    torch.manual_seed(7)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    w = torch.randn(shape[-1], device="cuda", dtype=dtype)
    rtol, atol = {torch.float32: (2e-4, 2e-5), torch.float16: (2e-3, 2e-3), torch.bfloat16: (2e-2, 2e-2)}[dtype]
    torch.testing.assert_close(rmsnorm(x, w, num_warps=warps), rmsnorm_reference(x, w), rtol=rtol, atol=atol)
    torch.testing.assert_close(softmax(x, num_warps=warps), softmax_reference(x), rtol=rtol, atol=atol)


def test_softmax_large_values():
    x = torch.tensor([[10000., -10000., 0.]], device="cuda")
    torch.testing.assert_close(softmax(x), softmax_reference(x))
