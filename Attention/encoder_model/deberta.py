import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops.layers.torch import Rearrange


def disentangled_self_attention(q: Tensor, k: Tensor, v: Tensor, qr: Tensor, kr: Tensor, dot_scale: Tensor) -> Tensor:
    """
    Disentangled Self-Attention for DeBERTa
    Args:
        q: content query matrix, shape (batch_size, seq_len, dim_head)
        k: content key matrix, shape (batch_size, seq_len, dim_head)
        v: content value matrix, shape (batch_size, seq_len, dim_head)
        qr: position query matrix, shape (batch_size, 2*max_relative_position, dim_head), r means relative position
        kr: position key matrix, shape (batch_size, 2*max_relative_position, dim_head), r means relative position
        dot_scale: scale factor for Q•K^T result, same as pure transformer
    Math:
        Attention Matrix = A_c2c + A_c2r + A_r2c
        A = softmax(Attention Matrix/sqrt(3*D_h)), SA(z) = Av
    References:
        https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/deberta/disentangled_attention.py
        https://arxiv.org/pdf/1803.02155.pdf

    """
    c2c = torch.matmul(q, k.transpose(-1, -2))  # A_c2c
    tmp_c2p = torch.stack(
        [torch.matmul(q[i, :], kr.transpose(-1, -2)) for i in range(q.shape[0])],
        dim=0
    )

    attention_dist = F.softmax(
        torch.matmul(q, k.transpose(-1, -2)) / dot_scale,
        dim=-1
    )
    attention_matrix = torch.matmul(attention_dist, v)
    return attention_matrix


class RelativePositionEmbedding(nn.Module):
    """
    In this class, we implement Relative Position Embedding Layer for DeBERTa Encoder
    Args:

    References:
        https://arxiv.org/abs/2006.03654
        https://arxiv.org/abs/2111.09543
        https://arxiv.org/abs/1901.02860
        https://arxiv.org/abs/1906.08237
    """


class AttentionHead(nn.Module):
    """
    In this class, we implement workflow of single attention head for DeBERTa
    Args:
        dim_model: dimension of model's latent vector space, default 1024 from official paper
        dim_head: dimension of each attention head, default 64 from official paper (1024 / 16)
        dropout: dropout rate, default 0.1
    Math:
        [q,k,v]=z•U_qkv, A = softmax(q•k^t/sqrt(D_h)), SA(z) = Av
    """
    def __init__(self, dim_model: int =  1024, dim_head: int = 64, dropout: float = 0.1) -> None:
        super(AttentionHead, self).__init__()
        self.dim_model = dim_model
        self.dim_head = dim_head
        self.dropout = dropout
        self.dot_scale = torch.sqrt(torch.tensor(3 * self.dim_head))  # from official paper by microsoft
        self.fc_q = nn.Linear(self.dim_model, self.dim_head)
        self.fc_k = nn.Linear(self.dim_model, self.dim_head)
        self.fc_v = nn.Linear(self.dim_model, self.dim_head)
        self.fc_qr = nn.Linear(self.dim_model, self.dim_head)  # projector for Relative Position Query matrix
        self.fc_kr = nn.Linear(self.dim_model, self.dim_head)  # projector for Relative Position Key matrix

    def forward(self, x: Tensor) -> Tensor:
        attention_matrix = disentangled_self_attention(
            self.fc_q(x),
            self.fc_k(x),
            self.fc_v(x),
            self.fc_qr(x),  # Relative Position Query matrix
            self.fc_kr(x),  # Relative Position Key matrix
            self.dot_scale
        )
        return attention_matrix


class MultiHeadAttention(nn.Module):
    """
    In this class, we implement workflow of Multi-Head Self-Attention for DeBERTa
    Args:
        dim_model: dimension of model's latent vector space, default 1024 from official paper
        num_heads: number of heads in MHSA, default 16 from official paper for ViT-Large
        dim_head: dimension of each attention head, default 64 from official paper (1024 / 16)
        dropout: dropout rate, default 0.1
    Math:
        MSA(z) = [SA1(z); SA2(z); · · · ; SAk(z)]•Umsa
    Reference:
        https://arxiv.org/abs/2010.11929
        https://arxiv.org/abs/1706.03762
        https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/deberta/bert.py
    """
    def __init__(self, dim_model: int = 1024, num_heads: int = 8, dim_head: int = 64, dropout: float = 0.1) -> None:
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
    Class for FeedForward module in ViT-Large
    Args:
        dim_model: dimension of model's latent vector space, default 512
        dim_mlp: dimension of FFN's hidden layer, default 2048 from official paper
        dropout: dropout rate, default 0.1
    Math:
        MLP(x) = MLP(LN(x))+x
    """
    def __init__(self, dim_model: int = 1024, dim_mlp: int = 4096, dropout: float = 0.1) -> None:
        super(FeedForward, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_model, dim_mlp),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim_mlp, dim_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class DeBERTaEncoderLayer(nn.Module):
    """
    Class for encoder_model module in ViT-Large
    In this class, we stack each encoder_model module (Multi-Head Attention, Residual-Connection, Layer Normalization, MLP)
    Args:

    References:
        https://arxiv.org/abs/2006.03654
        https://arxiv.org/abs/2111.09543
    """
    def __init__(self, dim_model: int = 1024, num_heads: int = 16, dim_mlp: int = 4096, dropout: float = 0.1) -> None:
        super(DeBERTaEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            dim_model,
            num_heads,
            int(dim_model / num_heads),
            dropout,
        )
        self.layer_norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(p=dropout)
        self.mlp = FeedForward(
            dim_model,
            dim_mlp,
            dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        ln_x = self.layer_norm(x)
        residual_x = self.dropout(self.self_attention(ln_x)) + x

        ln_x = self.layer_norm(residual_x)
        fx = self.layer_norm(self.mlp(ln_x) + residual_x)  # from official paper & code by Google Research
        return fx


class EMD(nn.Module):
    """
    Class for Enhanced Mask Decoder module in DeBERTa, which is used for Masked Language Model
    Word 'Decoder' means that denoise masked token by predicting masked token
    This module also same as basic deberta encoder layer but, add Absolute Position Embedding for MLM Task
    Args:

    References:
        https://arxiv.org/abs/2006.03654
        https://arxiv.org/abs/2111.09543
    """
    pass

