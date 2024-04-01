import math
import torch
import torch.nn as nn
from torch import Tensor


class LoRA(nn.Module):
    """ class module for Low-Rank adaptation of LLM SFT
    This module return result of "BAx*(a/r)" in mathematical expression in official paper

    Args:
        dim: dimension of input tensor
        rank: rank of tensor, which is hyperparameter for LoRA
        alpha: hyperparameter for LoRA, trainable parameter, which is initialized by rank value
        options: default str, 'rlora' which is already proved to work better than pure lora
                 you can select pure lora as passing argument 'lora'

    Math:
        h = W0x + ∆Wx = W0x + BAx*(a/r)

    Notes:
        we use sqrt(rank) value, it is already proven to work better in LoRA,
        from Huggingface PEFT library official docs

    References:
        https://arxiv.org/abs/2106.09685
        https://pytorch.org/blog/understanding-gpu-memory-1/
    """
    def __init__(self, dim: int, rank: int, alpha: int, options: str = 'rlora'):
        super().__init__()
        self.a = nn.Parameter(torch.randn(rank, dim))  # init by random Gaussian distribution (normal distribution)
        self.b = nn.Parameter(torch.zeros(dim, rank))  # init by zero
        self.alpha = alpha / math.sqrt(rank) if options == 'rlora' else alpha / rank

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.matmul(inputs, self.b @ self.a) * self.alpha

