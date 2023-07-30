import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops.layers.torch import Rearrange


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
    Math:
        MHA(Q, K, V) = Concat(Head1, Head2, ... Head8) * W_concat
    Reference:
        https://arxiv.org/abs/1706.03762
    """
    def __init__(self, dim_model: int = 512, num_heads: int = 8, dropout: float = 0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        self.dim = dim_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.dim_head = int(self.dim / self.num_heads)  # dimension of each attention head
        self.dot_scale = torch.sqrt(torch.tensor(self.dim_head))  # scale factor for Q•K^T Result

        # linear combination: projection matrix(Q_1, K_1, V_1, ... Q_n, K_n, V_n) for each attention head
        self.projector = Projector(self.num_heads, self.dim_head)  # init instance
        self.projector_list = [list(self.projector()) for _ in range(self.num_heads)]  # call instance
        self.fc_concat = nn.Linear(self.dim, self.dim)  # for concatenation of each attention head

    def forward(self, x: Tensor, mask: bool = None) -> Tensor:
        """
        1) make Q, K, V matrix for each attention head: [BS, HEAD, SEQ_LEN, DIM_HEAD], ex) [10, 8, 512, 64]
        2) Do self-attention in each attention head
            - Matmul (Q, K^T) with scale factor (sqrt(DIM_HEAD))
            - Mask for padding token (Option for Decoder)
            - Softmax
            - Matmul (Softmax, V)
        3) Concatenate each attention head & linear transformation (512, 512)
        """
        # 1) make Q, K, V matrix for each attention head
        Q, K, V = [], [], []

        for i in range(self.num_heads):
            Q.append(self.projector_list[i][0](x))
            K.append(self.projector_list[i][1](x))
            V.append(self.projector_list[i][2](x))

        Q = torch.stack(Q, dim=1)
        K = torch.stack(K, dim=1)
        V = torch.stack(V, dim=1)

        # 2) Do self-attention in each attention head
        attention_score = torch.matmul(Q, K.transpose(-1, -2)) / self.dot_scale
        if mask is not None:  # for padding token
            attention_score[mask] = float('-inf')
        attention_dist = F.softmax(attention_score, dim=-1)  # [BS, HEAD, SEQ_LEN, SEQ_LEN]
        attention_matrix = torch.matmul(attention_dist, V).transpose(1, 2).reshape(x.shape[0], x.shape[1], self.dim)

        # 3) Concatenate each attention head & linear transformation (512, 512)
        x = self.fc_concat(attention_matrix)
        return x


class FeedForward(nn.Module):
    """
    Class for feed-forward network (FFN) module in vanilla transformer
    We apply GELU, which is a variant of ReLU
    Args:
        dim_model: dimension of model's latent vector space, default 512
        dim_ffn: dimension of FFN's hidden layer, default 2048 from official paper
        dropout: dropout rate, default 0.1
    Math:
        FFN(x) = max(0, x*W_1 + b_1)*W_2 + b_2
    """
    def __init__(self, dim_model: int = 512, dim_ffn: int = 2048, dropout: float = 0.1) -> None:
        super(FeedForward, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_ffn),
            nn.GELU(),
            nn.Linear(dim_ffn, dim_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)


class EncoderLayer(nn.Module):
    """
    Class for encoder_model module in vanilla transformer
    In this class, we stack each encoder_model module (Multi-Head Attention, Residual-Connection, Layer Normalization, FFN)
    We apply post-layer Residual-Connection, which is different from original paper
    In common sense, post-layer Residual-Connection are more effective & stable than pre-layer Residual-Connection
    """
    def __init__(self, dim_model: int = 512, num_heads: int = 8, dim_ffn: int = 2048, dropout: float = 0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            dim_model,
            num_heads,
            dropout,
        )
        self.layer_norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForward(
            dim_model,
            dim_ffn,
            dropout
        )

    def forward(self, x: Tensor) -> Tensor:
        """ x is word embedding which is sum of input embedding & positional embedding """
        ln_x = self.layer_norm(x)
        residual_x = self.self_attention(ln_x) + x

        ln_x = self.layer_norm(residual_x)
        fx = self.ffn(ln_x) + residual_x
        return fx


class Encoder(nn.Module):
    """
    In this class, encode input sequence and then we stack N EncoderLayer
    First, we define "positional embedding" and then add to input embedding for making word embedding
    Second, forward word embedding to N EncoderLayer and then get output embedding
    Args:
        input_embedding: embedding from input sequence, shape => [BS, SEQ_LEN, DIM_MODEL]
        max_len: max length of input sequence
        N: number of EncoderLayer, default 6 for base model
    """
    def __init__(
        self,
        input_embedding: Tensor,
        max_len: int = 512,
        N: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_ffn: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super(Encoder, self).__init__()
        self.input_embedding = input_embedding
        self.scale = torch.sqrt(torch.Tensor(dim_model))  # scale factor for input embedding
        self.positional_embedding = nn.Embedding(max_len, dim_model)
        self.num_layers = N
        self.dim_model = dim_model
        self.max_len = max_len
        self.num_heads = num_heads
        self.dim_ffn = dim_ffn
        self.dropout = dropout
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(dim_model, num_heads, dim_ffn, dropout) for _ in range(self.num_layers)]
        )

    def forward(self, mask: Tensor) -> tuple[Tensor, Tensor]:
        """ forward function for Encoder """
        layer_output = []
        pos_x = torch.arange(self.max_len).repeat(self.input_embedding.shape[0]).to(self.input_embedding.device)
        x = self.scale * self.input_embedding + self.positional_embedding(pos_x)

        for layer in self.encoder_layers:
            x = layer(x)  # embedding vector from 1 encoder_model layer
            layer_output.append(x)
        layer_output = torch.stack(layer_output, dim=0).to(x.device)  # For Weighted Layer Pool: [N, BS, SEQ_LEN, DIM]
        return x, layer_output


