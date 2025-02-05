""" py module of MoE (Mixture of Experts) implementation for transformer architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple, List
from einops.layers.torch import Rearrange
from transformers import AutoConfig, AutoTokenizer
from experiment.models.abstract_model import AbstractModel


class Experts(nn.Module):
    """
    """
    def __init__(self, dim_model: int):
        super(Experts, self).__init__()
        self.expert = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.expert(x)


class SparseMoELayer(nn.Module):
    """ main module of MoE, replacing the dense feed-forward network in transformer encoder/decoder blocks.
    this module has the gating network, experts network

    Args:
        k (int): value of top-k routing
        dim_model(int): value of gating network's input size

    Reference:
        - https://huggingface.co/blog/moe
        - https://arxiv.org/pdf/2101.03961  # switch transformer paper
    """
    def __init__(self, k: int, num_experts: int, dim_model: int):
        super(SparseMoELayer, self).__init__()
        self.k = k  # value of top-k routing
        self.gate = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.Softmax(dim=-1)
        )
        self.experts = nn.ModuleList([Experts(dim_model) for _ in range(num_experts)])

    def forward(self, x: Tensor) -> None:
        gating = self.gate(x)
        """ need to implement top-k routing and sparse activation logic here
        """
        return


class MoE(nn.Module, AbstractModel):
    """ interface module of MoE(Mixture of Experts) for transformer architecture
    Args:

    Reference:
        - https://huggingface.co/blog/moe
        - https://arxiv.org/pdf/2101.03961  # switch transformer paper
    """
    def __init__(self, cfg):
        super(MoE, self).__init__()
        self.cfg = cfg
        self.dim_model = cfg.dim_model
        self.num_layers = cfg.num_layers
        self.num_experts = cfg.num_experts

    def forward(self):
        return


if __name__ == '__main__':
    pass