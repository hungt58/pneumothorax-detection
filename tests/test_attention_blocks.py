import torch
from models.attention_blocks import AttentionGate, AttentionUpBlock


def test_gate_and_up_block_gradients():
    skip = torch.randn(2, 8, 31, 35, requires_grad=True)
    gate_input = torch.randn(2, 16, 31, 35, requires_grad=True)
    gate = AttentionGate(8, 16)
    alpha = gate.attention(skip, gate_input)
    assert alpha.shape == (2, 1, 31, 35)
    assert torch.all((alpha >= 0) & (alpha <= 1))
    output = gate(skip, gate_input)
    assert output.shape == skip.shape
    output.mean().backward()
    assert skip.grad is not None and gate_input.grad is not None
    block = AttentionUpBlock(16, 8, 8)
    assert block(torch.randn(2, 16, 15, 17), skip.detach()).shape == skip.shape
