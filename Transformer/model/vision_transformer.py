import pandas as pd
import numpy as np
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
        dot_scale: scale factor for Q•K^T result, same as pure transformer
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
        self.dot_scale = torch.sqrt(torch.tensor(self.dim_head))
        self.fc_q = nn.Linear(self.dim_model, self.dim_head)
        self.fc_k = nn.Linear(self.dim_model, self.dim_head)
        self.fc_v = nn.Linear(self.dim_model, self.dim_head)

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
        dim_model: dimension of model's latent vector space, default 1024 from official paper
        num_heads: number of heads in MHSA, default 16 from official paper for ViT-Large
        dim_head: dimension of each attention head, default 64 from official paper (1024 / 16)
        dropout: dropout rate, default 0.1
    Math:
        MSA(z) = [SA1(z); SA2(z); · · · ; SAk(z)]•Umsa
    Reference:
        https://arxiv.org/abs/2010.11929
        https://arxiv.org/abs/1706.03762
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


class MLP(nn.Module):
    """
    Class for MLP module in ViT-Large
    Args:
        dim_model: dimension of model's latent vector space, default 512
        dim_mlp: dimension of FFN's hidden layer, default 2048 from official paper
        dropout: dropout rate, default 0.1
    Math:
        MLP(x) = MLP(LN(x))+x
    """
    def __init__(self, dim_model: int = 1024, dim_mlp: int = 4096, dropout: float = 0.1) -> None:
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_model, dim_mlp),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim_mlp, dim_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class VisionEncoderLayer(nn.Module):
    """
    Class for encoder module in ViT-Large
    In this class, we stack each encoder module (Multi-Head Attention, Residual-Connection, Layer Normalization, MLP)
    """
    def __init__(self, dim_model: int = 1024, num_heads: int = 16, dim_mlp: int = 4096, dropout: float = 0.1) -> None:
        super(VisionEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            dim_model,
            num_heads,
            int(dim_model / num_heads),
            dropout,
        )
        self.layer_norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(p=dropout)
        self.mlp = MLP(
            dim_model,
            dim_mlp,
            dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        ln_x = self.layer_norm(x)
        residual_x = self.dropout(self.self_attention(ln_x)) + x

        ln_x = self.layer_norm(residual_x)
        fx = self.mlp(ln_x) + residual_x
        return fx


class VisionEncoder(nn.Module):
    """
    In this class, encode input sequence(Image) and then we stack N VisionEncoderLayer
    First, we define "positional embedding" and then add to input embedding for making patch embedding
    Second, forward patch embedding to N EncoderLayer and then get output embedding
    Args:
        input_embedding: embedding from input sequence, shape => [BS, NUM_PATCH, DIM_MODEL]
        N: number of EncoderLayer, default 6 for base model
    """

    def __init__(self, input_embedding: Tensor, N: int = 24, dim_model: int = 1024, num_heads: int = 16, dim_mlp: int = 4096, dropout: float = 0.1) -> None:
        super(VisionEncoder, self).__init__()
        self.input_embedding = input_embedding
        self.scale = torch.sqrt(torch.Tensor(dim_model))  # scale factor for input embedding
        self.positional_embedding = nn.Embedding(self.input_embedding.shape[1], dim_model)
        self.num_layers = N
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_mlp = dim_mlp
        self.dropout = nn.Dropout(p=dropout)
        self.encoder_layers = nn.ModuleList(
            [VisionEncoderLayer(dim_model, num_heads, dim_mlp, dropout) for _ in range(self.num_layers)]
        )

    def forward(self) -> tuple[Tensor, Tensor]:
        layer_output = []
        pos_x = torch.arange(self.input_embedding.shape[1]).repeat(self.input_embedding.shape[0]).to(self.input_embedding.device)
        x = self.dropout(
            self.input_embedding + self.positional_embedding(pos_x)
        )
        for layer in self.encoder_layers:
            x = layer(x)
            layer_output.append(x)
        layer_output = torch.stack(layer_output, dim=0).to(x.device)  # For Weighted Layer Pool: [N, BS, SEQ_LEN, DIM]
        return x, layer_output


