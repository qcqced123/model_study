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

    Math:
        h = W0x + ∆Wx = W0x + BAx*(a/r)

    References:
        https://arxiv.org/abs/2106.09685
        https://pytorch.org/blog/understanding-gpu-memory-1/
    """
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.a = nn.Parameter(torch.randn(dim, rank))  # init by random Gaussian distribution (normal distribution)
        self.b = nn.Parameter(torch.zeros(rank, dim))  # init by zero
        self.alpha = nn.Parameter(torch.tensor(rank)) / rank

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.matmul(inputs, self.a) @ self.b * self.alpha

