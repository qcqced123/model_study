import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops.layers.torch import Rearrange


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, dot_scale: Tensor) -> Tensor:
    """
    Scaled Dot-Product Attention
    Args:
        q: query matrix, shape (batch_size, seq_len, dim_head)
        k: key matrix, shape (batch_size, seq_len, dim_head)
        v: value matrix, shape (batch_size, seq_len, dim_head)
        dot_scale: scale factor for Q•K^T result
    Math:
        A = softmax(q•k^t/sqrt(D_h)), SA(z) = Av
    """
    attention_dist = F.softmax(
        torch.matmul(q, k.transpose(-1, -2)) / dot_scale,
        dim=-1
    )
    attention_matrix = torch.matmul(attention_dist, v)
    return attention_matrix


class AttentionHead(nn.Module):
    """
    In this class, we implement workflow of single attention head
    Args:
        dim_model: dimension of model's latent vector space, default 512 from official paper
        dim_head: dimension of each attention head, default 64 from official paper (512 / 8)
        dropout: dropout rate, default 0.1
    Math:
        [q,k,v]=z•U_qkv, A = softmax(q•k^t/sqrt(D_h)), SA(z) = Av
    """
    def __init__(self, dim_model: int = 512, dim_head: int = 64, dropout: float = 0.1) -> None:
        super(AttentionHead, self).__init__()
        self.dim_model = dim_model
        self.dim_head = dim_head  # 512 / 8 = 64
        self.dropout = dropout
        self.dot_scale = torch.sqrt(torch.tensor(self.dim_head))
        self.fc_q = nn.Linear(self.dim_model, self.dim_head)  # Linear Projection for Query Matrix
        self.fc_k = nn.Linear(self.dim_model, self.dim_head)  # Linear Projection for Key Matrix
        self.fc_v = nn.Linear(self.dim_model, self.dim_head)  # Linear Projection for Value Matrix

    def forward(self, x: Tensor) -> Tensor:
        attention_matrix = scaled_dot_product_attention(
            self.fc_q(x),
            self.fc_k(x),
            self.fc_v(x),
            self.dot_scale
        )
        return attention_matrix


class MultiHeadAttention(nn.Module):
    """
    In this class, we implement workflow of Multi-Head Self-Attention
    Args:
        dim_model: dimension of model's latent vector space, default 512 from official paper
        num_heads: number of heads in MHSA, default 8 from official paper for Transformer
        dim_head: dimension of each attention head, default 64 from official paper (512 / 8)
        dropout: dropout rate, default 0.1
    Math:
        MSA(z) = [SA1(z); SA2(z); · · · ; SAk(z)]•Umsa
    Reference:
        https://arxiv.org/abs/1706.03762
    """
    def __init__(self, dim_model: int = 512, num_heads: int = 8, dim_head: int = 64, dropout: float = 0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.attention_heads = nn.ModuleList(
            [AttentionHead(self.dim_model, self.dim_head, self.dropout) for _ in range(self.num_heads)]
        )
        self.fc_concat = nn.Linear(self.dim_model, self.dim_model)

    def forward(self, x: Tensor) -> Tensor:
        """ x is already passed nn.Layernorm """
        assert x.ndim == 3, f'Expected (batch, seq, hidden) got {x.shape}'
        attention_output = self.fc_concat(
            torch.cat([head(x) for head in self.attention_heads], dim=-1)  # concat all dim_head = num_heads * dim_head
        )
        return attention_output


class FeedForward(nn.Module):
    """
    Class for Feed-Forward Network module in transformer
    In official paper, they use ReLU activation function, but GELU is better for now
    We change ReLU to GELU & add dropout layer
    Args:
        dim_model: dimension of model's latent vector space, default 512
        dim_ffn: dimension of FFN's hidden layer, default 2048 from official paper
        dropout: dropout rate, default 0.1
    Math:
        FeedForward(x) = FeedForward(LN(x))+x
    """
    def __init__(self, dim_model: int = 512, dim_ffn: int = 2048, dropout: float = 0.1) -> None:
        super(FeedForward, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_ffn),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim_ffn, dim_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)


