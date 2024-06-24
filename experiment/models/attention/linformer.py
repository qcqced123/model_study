import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from configuration import CFG
from typing import Tuple, List
from einops.layers.torch import Rearrange
from experiment.models.abstract_model import AbstractModel


def linear_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dot_scale: Tensor,
    attention_dropout: nn.Dropout,
    padding_mask: Tensor = None,
    attention_mask: Tensor = None
) -> Tensor:
    """ function for linear attention from linformer paper written by Facebook AI Research

    Math:
        A_i = softmax(QWq•E•KWk/sqrt(d))•F•VWv  (i: ith attention head)

    Args:
        each tensor must be already projected by each projector, especially for Key, Value Tensor

    Returns:

    Reference:
        https://arxiv.org/pdf/2006.04768

    """
    batch, num_heads, seq_len, dim_head = q.shape  # assign alias for each dimension
    attention_matrix = torch.matmul(q, k) / dot_scale

    if padding_mask is not None:
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # for broadcasting: shape (BS, 1, 1, SEQ_LEN)
        attention_matrix = attention_matrix.masked_fill(padding_mask == 1, float('-inf'))

    attention_dist = attention_dropout(
        F.softmax(attention_matrix, dim=-1)
    )
    attention_matrix = torch.matmul(attention_dist, v).permute(0, 2, 1, 3).view(
        batch,
        seq_len,
        num_heads*dim_head
    ).contiguous()
    return attention_matrix


class LinformerMultiHeadAttention(nn.Module):
    """ In this class, we implement workflow of Multi-Head Self-attention for Linformer

    Args:
        max_seq: maximum sequence length for setting the dimension of projector
        dim_model: dimension of model's latent vector space, default 1024 from official paper
        num_attention_heads: number of heads in MHSA, default 16 from official paper for Transformer
        dim_head: dimension of each attention head, default 64 from official paper (1024 / 16)
        low_rank: low-rank approximation for linear projection, default 128 from official paper
        attention_dropout_prob: dropout rate, default 0.1

    Math:
        A = softmax(attention Matrix/sqrt(3*D_h)), SA(z) = Av

    Reference:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/pdf/1810.04805.pdf
        https://arxiv.org/pdf/2006.04768
    """
    def __init__(
        self,
        max_seq: int = 512,
        dim_model: int = 1024,
        num_attention_heads: int = 16,
        dim_head: int = 64,
        low_rank: int = 128,
        attention_dropout_prob: float = 0.1
    ) -> None:
        super(LinformerMultiHeadAttention, self).__init__()
        self.max_seq = max_seq
        self.dim_model = dim_model
        self.num_attention_heads = num_attention_heads
        self.dim_head = dim_head
        self.low_rank = low_rank
        self.fc_q = nn.Linear(self.dim_model, self.dim_model)
        self.fc_k = nn.Linear(self.dim_model, self.dim_model)
        self.fc_v = nn.Linear(self.dim_model, self.dim_model)
        self.projector_e = nn.Linear(self.low_rank, self.max_seq, bias=False)
        self.projector_f = nn.Linear(self.low_rank, self.max_seq, bias=False)

        self.fc_concat = nn.Linear(self.dim_model, self.dim_model)

        self.attention = linear_attention
        self.dot_scale = torch.sqrt(torch.tensor(self.dim_head)).to('cuda')
        self.attention_dropout = nn.Dropout(p=attention_dropout_prob)

    def forward(self, x: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tensor:
        """ x is already passed nn.Layernorm """
        assert x.ndim == 3, f'Expected (batch, seq, hidden) got {x.shape}'

        # size: bs, seq, nums head, dim head, linear projection
        q = self.fc_q(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        k = self.projector_e(
            self.fc_k(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head).permute(0, 2, 3, 1).contiguous()
        )
        v = self.projector_f(
            self.fc_v(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        ).transpose(-1, -2)

        attention_matrix = self.attention(
            q,
            k,
            v,
            self.dot_scale,
            self.attention_dropout,
            padding_mask,
            attention_mask
        )
        attention_output = self.fc_concat(attention_matrix)
        return attention_output
