import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops.layers.torch import Rearrange


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, dot_scale: Tensor, mask: Tensor = None) -> Tensor:
    """
    Scaled Dot-Product attention with Masking for Decoder
    Args:
        q: query matrix, shape (batch_size, seq_len, dim_head)
        k: key matrix, shape (batch_size, seq_len, dim_head)
        v: value matrix, shape (batch_size, seq_len, dim_head)
        dot_scale: scale factor for Q•K^T result
        mask: there are three types of mask, mask matrix shape must be same as single attention head
              1) Encoder padded token
              2) Decoder Masked-Self-attention
              3) Decoder's Encoder-Decoder attention
    Math:
        A = softmax(q•k^t/sqrt(D_h)), SA(z) = Av
    """
    attention_matrix = torch.matmul(q, k.transpose(-1, -2)) / dot_scale
    if mask is not None:
        attention_matrix = attention_matrix.masked_fill(mask == 0, float('-inf'))
    attention_dist = F.softmax(attention_matrix, dim=-1)
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

    def forward(self, x: Tensor, mask: Tensor, enc_output: Tensor = None) -> Tensor:
        q, k, v = self.fc_q(x), self.fc_k(x), self.fc_v(x)  # x is previous layer's output
        if enc_output is not None:
            """ For encoder-decoder self-attention """
            k = self.fc_k(enc_output)
            v = self.fc_v(enc_output)
        attention_matrix = scaled_dot_product_attention(q, k, v, self.dot_scale, mask=mask)
        return attention_matrix


class MultiHeadAttention(nn.Module):
    """
    In this class, we implement workflow of Multi-Head Self-attention
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

    def forward(self, x: Tensor, mask: Tensor, enc_output: Tensor = None) -> Tensor:
        """ x is already passed nn.Layernorm """
        assert x.ndim == 3, f'Expected (batch, seq, hidden) got {x.shape}'
        attention_output = self.fc_concat(
            torch.cat([head(x, mask, enc_output) for head in self.attention_heads], dim=-1)
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
    In this class, we stack each encoder_model module (Multi-Head attention, Residual-Connection, LayerNorm, FFN)
    We apply pre-layernorm, which is different from original paper
    In common sense, pre-layernorm are more effective & stable than post-layernorm
    """
    def __init__(self, dim_model: int = 512, num_heads: int = 8, dim_ffn: int = 2048, dropout: float = 0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            dim_model,
            num_heads,
            int(dim_model / num_heads),
            dropout,
        )
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(p=dropout)
        self.ffn = FeedForward(
            dim_model,
            dim_ffn,
            dropout,
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        ln_x = self.layer_norm1(x)
        residual_x = self.dropout(self.self_attention(ln_x, mask)) + x

        ln_x = self.layer_norm2(residual_x)
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
    def __init__(self, max_seq: 512, N: int = 6, dim_model: int = 512, num_heads: int = 8, dim_ffn: int = 2048, dropout: float = 0.1) -> None:
        super(Encoder, self).__init__()
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

    def forward(self, inputs: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        inputs: embedding from input sequence, shape => [BS, SEQ_LEN, DIM_MODEL]
        mask: mask for Encoder padded token for speeding up to calculate attention score
        """
        layer_output = []
        pos_x = torch.arange(self.max_seq).repeat(inputs.shape[0]).to(inputs)
        x = self.dropout(
            self.scale * inputs + self.positional_embedding(pos_x)
        )
        for layer in self.encoder_layers:
            x = layer(x, mask)
            layer_output.append(x)
        encoded_x = self.layer_norm(x)  # from official paper & code by Google Research
        layer_output = torch.stack(layer_output, dim=0).to(x.device)  # For Weighted Layer Pool: [N, BS, SEQ_LEN, DIM]
        return encoded_x, layer_output


class DecoderLayer(nn.Module):
    """
    Class for decoder model module in Transformer
    In this class, we stack each decoder_model module (Masked Multi-Head attention, Residual-Connection, LayerNorm, FFN)
    We apply pre-layernorm, which is different from original paper
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
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(p=dropout)  # dropout is not learnable layer

        self.ffn = FeedForward(
            dim_model,
            dim_ffn,
            dropout,
        )

    def forward(self, x: Tensor, dec_mask: Tensor, enc_dec_mask: Tensor, enc_output: Tensor) -> Tensor:
        ln_x = self.layer_norm1(x)
        residual_x = self.dropout(self.masked_attention(ln_x, dec_mask)) + x

        ln_x = self.layer_norm2(residual_x)
        residual_x = self.dropout(self.enc_dec_attention(ln_x, enc_dec_mask, enc_output)) + x  # for enc_dec self-attention

        ln_x = self.layer_norm3(residual_x)
        fx = self.ffn(ln_x) + residual_x
        return fx


class Decoder(nn.Module):
    """
    In this class, decode encoded embedding from encoder by outputs (target language, Decoder's Input Sequence)
    First, we define "positional embedding" for Decoder's Input Sequence,
    and then add them to Decoder's Input Sequence for making "decoder word embedding"
    Second, forward "decoder word embedding" to N DecoderLayer and then pass to linear & softmax for OutPut Probability
    Args:
        vocab_size: size of vocabulary for output probability
        max_seq: maximum sequence length, default 512 from official paper
        N: number of EncoderLayer, default 6 for base model
    References:
        https://arxiv.org/abs/1706.03762
    """
    def __init__(
        self,
        vocab_size: int,
        max_seq: int = 512,
        N: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_ffn: int = 2048,
        dropout: float = 0.1
    ) -> None:
        super(Decoder, self).__init__()
        self.max_seq = max_seq
        self.scale = torch.sqrt(torch.Tensor(dim_model))  # scale factor for input embedding from official paper
        self.positional_embedding = nn.Embedding(max_seq, dim_model)  # add 1 for cls token
        self.num_layers = N
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_ffn = dim_ffn
        self.dropout = nn.Dropout(p=dropout)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(dim_model, num_heads, dim_ffn, dropout) for _ in range(self.num_layers)]
        )
        self.layer_norm = nn.LayerNorm(dim_model)
        self.fc_out = nn.Linear(dim_model, vocab_size)  # In Pytorch, nn.CrossEntropyLoss already has softmax function

    def forward(self, inputs: Tensor, dec_mask: Tensor, enc_dec_mask: Tensor, enc_output: Tensor) -> tuple[Tensor, Tensor]:
        """
        inputs: embedding from input sequence, shape => [BS, SEQ_LEN, DIM_MODEL]
        dec_mask: mask for Decoder padded token for Language Modeling
        enc_dec_mask: mask for Encoder-Decoder Self-attention, from encoder padded token
        """
        layer_output = []
        pos_x = torch.arange(self.max_seq).repeat(inputs.shape[0]).to(inputs)
        x = self.dropout(
            self.scale * inputs + self.positional_embedding(pos_x)
        )
        for layer in self.decoder_layers:
            x = layer(x, dec_mask, enc_dec_mask, enc_output)
            layer_output.append(x)
        decoded_x = self.fc_out(self.layer_norm(x))  # Because of pre-layernorm
        layer_output = torch.stack(layer_output, dim=0).to(x.device)  # For Weighted Layer Pool: [N, BS, SEQ_LEN, DIM]
        return decoded_x, layer_output


class Transformer(nn.Module):
    """
    Main class for Pure Transformer, Pytorch implementation
    There are two Masking Method for padding token
        1) Row & Column masking
        2) Column masking only at forward time, Row masking at calculating losses time
    second method is more efficient than first method, first method is complex & difficult to implement
    Args:
        enc_vocab_size: size of vocabulary for Encoder Input Sequence
        dec_vocab_size: size of vocabulary for Decoder Input Sequence
        max_seq: maximum sequence length, default 512 from official paper
        enc_N: number of EncoderLayer, default 6 for base model
        dec_N: number of DecoderLayer, default 6 for base model
    Reference:
        https://arxiv.org/abs/1706.03762
    """
    def __init__(
        self,
        enc_vocab_size: int,
        dec_vocab_size: int,
        max_seq: int = 512,
        enc_N: int = 6,
        dec_N: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_ffn: int = 2048,
        dropout: float = 0.1
    ) -> None:
        super(Transformer, self).__init__()
        self.enc_input_embedding = nn.Embedding(enc_vocab_size, dim_model)
        self.dec_input_embedding = nn.Embedding(dec_vocab_size, dim_model)
        self.encoder = Encoder(max_seq, enc_N, dim_model, num_heads, dim_ffn, dropout)
        self.decoder = Decoder(dec_vocab_size, max_seq, dec_N, dim_model, num_heads, dim_ffn, dropout)

    @staticmethod
    def enc_masking(x: Tensor, enc_pad_index: int) -> Tensor:
        """ make masking matrix for Encoder Padding Token """
        enc_mask = (x != enc_pad_index).int().repeat(1, x.shape[-1]).view(x.shape[0], x.shape[-1], x.shape[-1])
        return enc_mask

    @staticmethod
    def dec_masking(x: Tensor, dec_pad_index: int) -> Tensor:
        """ make masking matrix for Decoder Masked Multi-Head Self-attention """
        pad_mask = (x != dec_pad_index).int().repeat(1, x.shape[-1]).view(x.shape[0], x.shape[-1], x.shape[-1])
        lm_mask = torch.tril(torch.ones(x.shape[0], x.shape[-1], x.shape[-1]))
        dec_mask = pad_mask * lm_mask
        return dec_mask

    @staticmethod
    def enc_dec_masking(enc_x: Tensor, dec_x: Tensor, enc_pad_index: int) -> Tensor:
        """ make masking matrix for Encoder-Decoder Multi-Head Self-attention in Decoder """
        enc_dec_mask = (enc_x != enc_pad_index).int().repeat(1, dec_x.shape[-1]).view(
            enc_x.shape[0], dec_x.shape[-1], enc_x.shape[-1]
        )
        return enc_dec_mask

    def forward(self, enc_inputs: Tensor, dec_inputs: Tensor, enc_pad_index: int, dec_pad_index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        enc_mask = self.enc_masking(enc_inputs, enc_pad_index)  # enc_x.shape[1] == encoder input sequence length
        dec_mask = self.dec_masking(dec_inputs, dec_pad_index)  # dec_x.shape[1] == decoder input sequence length
        enc_dec_mask = self.enc_dec_masking(enc_inputs, dec_inputs, enc_pad_index)

        enc_x, dec_x = self.enc_input_embedding(enc_inputs), self.dec_input_embedding(dec_inputs)

        enc_output, enc_layer_output = self.encoder(enc_x, enc_mask)
        dec_output, dec_layer_output = self.decoder(dec_x, dec_mask, enc_dec_mask, enc_output)
        return enc_output, dec_output, enc_layer_output, dec_layer_output
