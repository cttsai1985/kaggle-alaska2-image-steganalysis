"""CUDA availability check: one tiny bf16 forward/backward batch; not training.

Deselect with ``-m "not cuda"``.
"""

import pytest
import torch

pytestmark = pytest.mark.cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_one_batch_bf16_forward_backward_on_cuda(fixture_cfg):
    from alaska2.model import build_model

    model, _ = build_model(fixture_cfg(), pretrained=False)
    dev = torch.device("cuda", 0)
    model = model.to(dev).to(memory_format=torch.channels_last).train()
    x = torch.randn(4, 3, 64, 64, device=dev).contiguous(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x).float().logsumexp(1).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.cuda.max_memory_reserved(dev) < 2 * 2**30
