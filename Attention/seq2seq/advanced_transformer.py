import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops.layers.torch import Rearrange


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, dot_scale: Tensor, mask: Tensor = None) -> Tensor:
    """
    Scaled Dot-Product Attention with Masking for Decoder
    Args:
        q: query matrix, shape (batch_size, seq_len, dim_head)
        k: key matrix, shape (batch_size, seq_len, dim_head)
        v: value matrix, shape (batch_size, seq_len, dim_head)
        dot_scale: scale factor for Q•K^T result
        mask: mask for Encoder padded token & Decoder Masked-Self-Attention
    Math:
        A = softmax(q•k^t/sqrt(D_h)), SA(z) = Av
    """
    attention_dist = F.softmax(
        torch.matmul(q, k.transpose(-1, -2)) / dot_scale,
        dim=-1
    )
    if mask is not None:
        attention_dist = attention_dist.masked_fill(mask == 0, float('-inf'))
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

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        attention_matrix = scaled_dot_product_attention(
            self.fc_q(x),
            self.fc_k(x),
            self.fc_v(x),
            self.dot_scale,
            mask=mask
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

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """ x is already passed nn.Layernorm """
        assert x.ndim == 3, f'Expected (batch, seq, hidden) got {x.shape}'
        attention_output = self.fc_concat(
            torch.cat([head(x, mask) for head in self.attention_heads], dim=-1)
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


class EncoderLayer(nn.Module):
    """
    Class for encoder model module in Transformer
    In this class, we stack each encoder_model module (Multi-Head Attention, Residual-Connection, LayerNorm, FFN)
    We apply post-layer Residual-Connection, which is different from original paper
    In common sense, post-layer Residual-Connection are more effective & stable than pre-layer Residual-Connection
    """
    def __init__(self, dim_model: int = 512, num_heads: int = 8, dim_ffn: int = 2048, dropout: float = 0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            dim_model,
            num_heads,
            int(dim_model / num_heads),
            dropout,
        )
        self.layer_norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(p=dropout)
        self.ffn = FeedForward(
            dim_model,
            dim_ffn,
            dropout,
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        ln_x = self.layer_norm(x)
        residual_x = self.dropout(self.self_attention(ln_x, mask)) + x

        ln_x = self.layer_norm(residual_x)
        fx = self.ffn(ln_x) + residual_x
        return fx


class Encoder(nn.Module):
    """
    In this class, encode input sequence and then we stack N EncoderLayer
    First, we define "positional embedding" and then add to input embedding for making "word embedding"
    Second, forward "word embedding" to N EncoderLayer and then get output embedding
    In official paper, they use positional encoding, which is base on sinusoidal function(fixed, not learnable)
    But we use "positional embedding" which is learnable from training
    Args:
        max_seq: maximum sequence length, default 512 from official paper
        N: number of EncoderLayer, default 6 for base model
    """
    def __init__(self, mask: Tensor, max_seq: 512, N: int = 6, dim_model: int = 512, num_heads: int = 8, dim_ffn: int = 2048, dropout: float = 0.1) -> None:
        super(Encoder, self).__init__()
        self.mask = mask  # for encoder padding mask
        self.max_seq = max_seq
        self.scale = torch.sqrt(torch.Tensor(dim_model))  # scale factor for input embedding from official paper
        self.positional_embedding = nn.Embedding(max_seq, dim_model)  # add 1 for cls token
        self.num_layers = N
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_ffn = dim_ffn
        self.dropout = nn.Dropout(p=dropout)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(dim_model, num_heads, dim_ffn, dropout) for _ in range(self.num_layers)]
        )
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """ inputs: embedding from input sequence, shape => [BS, SEQ_LEN, DIM_MODEL] """
        layer_output = []
        pos_x = torch.arange(self.max_seq).repeat(inputs.shape[0]).to(inputs)
        x = self.dropout(
            self.scale * inputs + self.positional_embedding(pos_x)
        )
        for layer in self.encoder_layers:
            x = layer(x, self.mask)
            layer_output.append(self.layer_norm(x))
        encoded_x = self.layer_norm(x)  # from official paper & code by Google Research
        layer_output = torch.stack(layer_output, dim=0).to(x.device)  # For Weighted Layer Pool: [N, BS, SEQ_LEN, DIM]
        return encoded_x, layer_output


class DecoderLayer(nn.Module):
    """
    Class for decoder model module in Transformer
    In this class, we stack each decoder_model module (Masked Multi-Head Attention, Residual-Connection, LayerNorm, FFN)
    We apply post-layer Residual-Connection, which is different from original paper
    Args:
        Encoder's Output을 매 레이어 마다 받아야 하나..?
    References:
        https://arxiv.org/abs/1706.03762
    """
    def __init__(self, dim_model: int = 512, num_heads: int = 8, dim_ffn: int = 2048, dropout: float = 0.1) -> None:
        super(DecoderLayer, self).__init__()
        self.masked_attention = MultiHeadAttention(
            dim_model,
            num_heads,
            int(dim_model / num_heads),
            dropout,
        )
        self.enc_dec_attention = MultiHeadAttention(
            dim_model,
            num_heads,
            int(dim_model / num_heads),
            dropout,
        )
        self.layer_norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(p=dropout)
        self.ffn = FeedForward(
            dim_model,
            dim_ffn,
            dropout,
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        ln_x = self.layer_norm(x)
        residual_x = self.dropout(self.self_attention(ln_x, mask)) + x

        ln_x = self.layer_norm(residual_x)
        fx = self.ffn(ln_x) + residual_x
        return fx


class Decoder(nn.Module):
    pass


class Transformer(nn.Module):
    pass

