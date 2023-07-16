import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """
    Making projection matrix(Q, K, V) for each attention head
    When you call this class, it returns projection matrix of each attention head
    For example, if you call this class with 8 heads, it returns 8 set of projection matrices (Q, K, V)
    Args:
        num_heads: number of heads in MHA, default 8
        dim_head: dimension of each attention head, default 64
    """
    def __init__(self, num_heads: int = 8, dim_head: int = 64) -> None:
        super(Projector, self).__init__()
        self.dim_model = num_heads * dim_head
        self.num_heads = num_heads
        self.dim_head = dim_head

    def __call__(self):
        fc_q = nn.Linear(self.dim_model, self.dim_head)
        fc_k = nn.Linear(self.dim_model, self.dim_head)
        fc_v = nn.Linear(self.dim_model, self.dim_head)
        return fc_q, fc_k, fc_v


class MultiHeadAttention(nn.Module):
    """
    Class for multi-head attention (MHA) module in vanilla transformer
    We apply linear transformation to input vector by each attention head's projection matrix (8, 512, 64)
    Other approaches are possible, such as using one projection matrix for all attention heads (1, 512, 512)
    and then split into each attention heads (8. 512, 64)
    Args:
        dim_model: dimension of model's latent vector space, default 512 from official paper
        num_heads: number of heads in MHA, default 8 from official paper
        dropout: dropout rate, default 0.1
    Reference:
        https://arxiv.org/abs/1706.03762
    """
    def __init__(self, dim_model: int = 512, num_heads: int = 8, dropout: float = 0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        # hyper-parameters for model architecture
        self.dim = dim_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.dim_head = int(self.dim / self.num_heads)  # dimension of each attention head
        self.dot_scale = torch.sqrt(torch.tensor(self.dim_head))  # scale factor for Q•K^T Result

        # linear combination: projection matrix(Q_1, K_1, V_1, ... Q_n, K_n, V_n) for each attention head
        self.projector = [[Projector(self.num_heads, self.dim_head)] for _ in range(self.num_heads)]
        # self.fc_q = nn.Linear(self.dim, self.dim)  # for query matrix
        # self.fc_k = nn.Linear(self.dim, self.dim)  # for key matrix
        # self.fc_v = nn.Linear(self.dim, self.dim)  # for value matrix
        self.fc_concat = nn.Linear(self.dim, self.dim)  # for concatenation of each attention head

    def forward(self):
        pass
