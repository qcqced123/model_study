import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment.models.abstract_model import AbstractModel
from torch import Tensor
from typing import Tuple, List
from einops.layers.torch import Rearrange
from configuration import CFG


def global_attention():
    pass


def sliding_window_attention():
    pass


def dilated_sliding_window_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dot_scale: Tensor,
    attention_dropout: nn.Dropout,
    padding_mask: Tensor = None,
    attention_mask: Tensor = None
) -> Tensor:
    """ dilated sliding window attention from longformer

    Args:
        q: query matrix, shape (batch_size, seq_len, dim_head)
        k: key matrix, shape (batch_size, seq_len, dim_head)
        v: value matrix, shape (batch_size, seq_len, dim_head)
        dot_scale: scale factor for Q•K^T result
        attention_dropout: dropout for attention matrix, default rate is 0.1 from official paper
        padding_mask: mask for attention matrix for MLM, you must check whether or not padding token is 1
        attention_mask: mask for attention matrix for CLM

    Math:
        A = softmax(q•k^t/sqrt(D_h)), SA(z) = Av

    References:

    """
    pass


class MultiHeadAttention(nn.Module):
    """ In this class, we implement workflow of Multi-Head Self-attention for BERT
    This class has same role as Module "BertAttention" in official Repo (bert.py)
    In official repo, they use post-layer norm, but we use pre-layer norm which is more stable & efficient for training

    Args:
        dim_model: dimension of model's latent vector space, default 1024 from official paper
        num_attention_heads: number of heads in MHSA, default 16 from official paper for Transformer
        dim_head: dimension of each attention head, default 64 from official paper (1024 / 16)
        attention_dropout_prob: dropout rate, default 0.1

    Math:
        A = softmax(attention Matrix/sqrt(3*D_h)), SA(z) = Av

    Reference:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/pdf/1810.04805.pdf
    """
    def __init__(self, dim_model: int = 1024, num_attention_heads: int = 16, dim_head: int = 64, attention_dropout_prob: float = 0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        self.dim_model = dim_model
        self.num_attention_heads = num_attention_heads
        self.dim_head = dim_head
        self.fc_q = nn.Linear(self.dim_model, self.dim_model)
        self.fc_k = nn.Linear(self.dim_model, self.dim_model)
        self.fc_v = nn.Linear(self.dim_model, self.dim_model)
        self.fc_concat = nn.Linear(self.dim_model, self.dim_model)
        self.attention = dilated_sliding_window_attention
        self.dot_scale = torch.sqrt(torch.tensor(self.dim_head)).to('cuda')
        self.attention_dropout = nn.Dropout(p=attention_dropout_prob)

    def forward(self, x: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tensor:
        """ x is already passed nn.Layernorm """
        assert x.ndim == 3, f'Expected (batch, seq, hidden) got {x.shape}'

        # size: bs, seq, nums head, dim head, linear projection
        q = self.fc_q(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        k = self.fc_k(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head)
        v = self.fc_v(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

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


class FeedForward(nn.Module):
    """ Class for Feed-Forward Network module in Transformer Encoder Block, this module for BERT
    Same role as Module "BertIntermediate" in official Repo (bert.py)

    Args:
        dim_model: dimension of model's latent vector space, default 1024
        dim_ffn: dimension of FFN's hidden layer, default 4096 from official paper
        hidden_dropout_prob: dropout rate, default 0.1

    Math:
        FeedForward(x) = FeedForward(LN(x))+x
    """
    def __init__(self, dim_model: int = 1024, dim_ffn: int = 4096, hidden_dropout_prob: float = 0.1) -> None:
        super(FeedForward, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_ffn),
            nn.GELU(),
            nn.Dropout(p=hidden_dropout_prob),
            nn.Linear(dim_ffn, dim_model),
            nn.Dropout(p=hidden_dropout_prob),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)



