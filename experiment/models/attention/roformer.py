import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment.models.abstract_model import AbstractModel
from torch import Tensor
from typing import Tuple, Optional
from einops.layers.torch import Rearrange
from configuration import CFG


def apply_rotary_position_embeddings(sinusoidal_pos: Tensor, query_layer: Tensor, key_layer: Tensor, value_layer: Tensor = None):
    """ Apply rotary position encoding to query, key layer
    Original Source code from Huggingface's RoFormer model, which is the most optimized way to create positional embedding

    You can find mathematical proof in official paper's Appendix

    To tell you the truth, you doesn't need to match the shape of query, key exactly,
    but you must match the last dimension: dim_head

    Args:
        sinusoidal_pos: sinusoidal positional encoding, shape [batch(None), num_dim(None), seq_len, dim_head]
        query_layer: query matrix, shape (batch_size, num_head, seq_len, dim_head)
        key_layer: key matrix, shape (batch_size, num_head, seq_len, dim_head)
        value_layer: value matrix, shape (batch_size, num_head, seq_len, dim_head)

    References:
        https://arxiv.org/abs/2104.09864  # RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py#L323
    """
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)  # select two element of index values
    sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)

    cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
    rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
        query_layer
    )

    # mathematical expression from Appendix in official repo
    query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
    rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
    key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos

    if value_layer is not None:  # In official, they don't use value_layer
        rotate_half_value_layer = torch.stack([-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1).reshape_as(
            value_layer
        )
        value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
        return query_layer, key_layer, value_layer
    return query_layer, key_layer


def kernel_fn(x: Tensor, kernel_name: str) -> Tensor:
    """ Select kernel function for attention head
    This is temporary function, we will implement more kernel function in future
    """
    hidden_state = None
    if kernel_name == 'elu':
        hidden_state = F.elu(x) + 1
    return hidden_state


def linear_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kernel: str = 'elu',
    eps: float = 1e-6,
    attention_dropout: nn.Dropout = None,
    padding_mask: Tensor = None,
    attention_mask: Tensor = None,
) -> Tensor:
    """ Linear attention with masking for padding token
    This function is designed for parallel computation with head dimension, not using loop & concatenate method

    Args:
        q: query matrix, shape (batch_size, seq_len, dim_head)
        k: key matrix, shape (batch_size, seq_len, dim_head)
        v: value matrix, shape (batch_size, seq_len, dim_head)
        dim_head: default 64 (int), dimension of each attention head
        kernel: default elu (str), which is used in original paper
        eps: default 1e-8 (float), for numerical stability
        attention_dropout: default rate is 0.1, dropout for attention matrix
        padding_mask: mask for attention matrix for MLM, you must check whether or not padding token is 1
        attention_mask: mask for attention matrix for CLM

    Math:
        A = normalize(Φ(Q).mm(Φ(K).t())).mm(V)

    Einsum:
        b: batch_size
        s: sequence length of query
        h: number of heads
        q: dimension size of each query's heads
        k: dimension size of each key's heads
        v: dimension size of each value's heads

    References:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/pdf/1810.04805.pdf
        https://arxiv.org/abs/2006.16236
        https://arxiv.org/abs/2104.09864  # RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py
        https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
    """
    BS, SEQ_LEN, NUM_HEADS, DIM_HEADS = q.shape
    projected_q, projected_k = kernel_fn(q, kernel), kernel_fn(k, kernel)

    if padding_mask is not None:  # applying padding mask, calculating normalizer
        projected_k[padding_mask == 1] = 0

    projected_k = projected_k.permute(0, 2, 1, 3).contiguous()
    kv = torch.matmul(v.permute(0, 2, 3, 1).contiguous(), projected_k)
    z = 1 / torch.clamp(
        torch.mul(projected_q.permute(0, 2, 1, 3).contiguous(), projected_k.sum(dim=2).unsqueeze(2)).sum(dim=-1),
        min=eps)  # breakdown by sequence length dimension
    attention_matrix = attention_dropout(
        torch.einsum("bshq,bhvk,bhs->bshv", projected_q, kv, z).reshape(-1, SEQ_LEN, NUM_HEADS * DIM_HEADS)
    )
    return attention_matrix


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dot_scale: Tensor,
    attention_dropout: nn.Dropout,
    padding_mask: Tensor = None,
    attention_mask: Tensor = None
) -> Tensor:
    """ Scaled Dot-Product attention with Masking for padding mask, parallel version for Multi-Head Attention

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

    Reference:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/pdf/1810.04805.pdf
        https://arxiv.org/abs/2104.09864  # RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py
    """
    BS, NUM_HEADS, SEQ_LEN, DIM_HEADS = q.shape
    attention_matrix = torch.matmul(q, k.permute(0, 2, 3, 1).contiguous()) / dot_scale
    if padding_mask is not None:
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # for broadcasting: shape (BS, 1, 1, SEQ_LEN)
        attention_matrix = attention_matrix.masked_fill(padding_mask == 1, float('-inf'))

    attention_dist = attention_dropout(
        F.softmax(attention_matrix, dim=-1)
    )
    attention_matrix = torch.matmul(attention_dist, v).permute(0, 2, 1, 3).reshape(-1, SEQ_LEN, NUM_HEADS*DIM_HEADS).contiguous()
    return attention_matrix


class MultiHeadAttention(nn.Module):
    """ In this class, we implement workflow of Multi-Head Self-attention for Linear Transformers
    This class has same role as Module "BertAttention" in official Repo (bert.py)
    In official repo, they use post-layer norm, but we use pre-layer norm which is more stable & efficient for training

    Args:
        dim_model: dimension of model's latent vector space, default 1024 from official paper
        num_attention_heads: number of heads in MHSA, default 16 from official paper for Transformer
        dim_head: dimension of each attention head, default 64 from official paper (1024 / 16)
        kernel: kernel function for attention head, default 'elu' from official paper, softmax is also available
        attention_dropout_prob: dropout rate, default 0.1

    Math:
        A = softmax(attention Matrix/sqrt(3*D_h)), SA(z) = Av

    References:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/pdf/1810.04805.pdf
        https://arxiv.org/abs/2006.16236
        https://arxiv.org/abs/2104.09864  # RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py
        https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
    """

    def __init__(
        self,
        dim_model: int = 1024,
        num_attention_heads: int = 16,
        dim_head: int = 64,
        kernel: str = 'softmax',
        attention_dropout_prob: float = 0.1
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.dim_model = dim_model
        self.num_attention_heads = num_attention_heads
        self.dim_head = dim_head
        self.fc_q = nn.Linear(self.dim_model, self.dim_model)
        self.fc_k = nn.Linear(self.dim_model, self.dim_model)
        self.fc_v = nn.Linear(self.dim_model, self.dim_model)
        self.fc_concat = nn.Linear(self.dim_model, self.dim_model)
        self.apply_rope = apply_rotary_position_embeddings
        self.attention = scaled_dot_product_attention if kernel == 'softmax' else linear_attention
        self.attention_dropout = nn.Dropout(p=attention_dropout_prob)
        self.dot_scale = torch.sqrt(torch.tensor(self.dim_head, dtype=torch.float32))
        self.kernel = kernel
        self.eps = 1e-6

    # @staticmethod
    # def apply_rotary_position_embeddings(word: Tensor, rotary_pos: Tensor) -> Tensor:
    #     """ Very Un-Optimized way to apply rotary position encoding to word embedding
    #     Notes:
    #          ASAP, we will implement more optimized way to apply rotary position encoding to word embedding
    #     """
    #     BATCH_SIZE, SEQ_LEN, DIM_MODEL = word.shape
    #     result = torch.vstack([torch.bmm(rotary_pos, word[i].unsqueeze(-1)).squeeze(-1).view(SEQ_LEN, DIM_MODEL) for i in range(BATCH_SIZE)]).view(BATCH_SIZE, SEQ_LEN, DIM_MODEL)
    #     return result

    def forward(self, x: Tensor, rotary_pos_enc: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tensor:
        """ x is already passed nn.Layernorm, already multiplied with rotary position encoding """
        assert x.ndim == 3, f'Expected (batch, seq, hidden) got {x.shape}'

        # size: bs, seq, nums head, dim head, linear projection
        q = self.fc_q(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head)
        k = self.fc_k(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head)
        v = self.fc_v(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head)

        # multiple word embedding, rotary position encoding
        rotary_q, rotary_k = self.apply_rope(rotary_pos_enc, q, k)

        attention_matrix = None
        if self.kernel == 'elu':
            attention_matrix = self.attention(
                rotary_q,
                rotary_k,
                v,
                self.kernel,
                self.eps,
                self.attention_dropout,
                padding_mask,
                attention_mask
            )
        elif self.kernel == 'softmax':  # pure self-attention
            attention_matrix = self.attention(
                rotary_q.permute(0, 2, 1, 3).contiguous(),
                rotary_k,
                v.permute(0, 2, 1, 3).contiguous(),
                self.dot_scale,
                self.attention_dropout,
                padding_mask,
                attention_mask
            )

        attention_output = self.fc_concat(attention_matrix)
        return attention_output


class FeedForward(nn.Module):
    """ Class for Feed-Forward Network module in Transformer Encoder Block, this module for Linear Transformer
    Same role as Module "BertIntermediate" in official Repo (bert.py)

    Args:
        dim_model: dimension of model's latent vector space, default 1024
        dim_ffn: dimension of FFN's hidden layer, default 4096 from official paper
        hidden_dropout_prob: dropout rate, default 0.1

    Math:
        FeedForward(x) = FeedForward(LN(x))+x

    References:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/pdf/1810.04805.pdf
        https://arxiv.org/abs/2006.16236
        https://arxiv.org/abs/2104.09864  # RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py
        https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
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


class RoformerEncoderLayer(nn.Module):
    """ Class for encoder model module in Linear Transformer
    In this class, we stack each encoder_model module (Multi-Head attention, Residual-Connection, LayerNorm, FFN)
    This class has same role as Module "BertEncoder" in official Repo (bert.py)

    References:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/pdf/1810.04805.pdf
        https://arxiv.org/abs/2006.16236
        https://arxiv.org/abs/2104.09864  # RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py
        https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
    """
    def __init__(
            self,
            dim_model: int = 1024,
            num_attention_heads: int = 16,
            dim_ffn: int = 4096,
            kernel: str = 'elu',
            layer_norm_eps: float = 0.02,
            attention_dropout_prob: float = 0.1,
            hidden_dropout_prob: float = 0.1
    ) -> None:
        super(RoformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            dim_model,
            num_attention_heads,
            int(dim_model / num_attention_heads),
            kernel,
            attention_dropout_prob,
        )
        self.layer_norm1 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.hidden_dropout = nn.Dropout(p=hidden_dropout_prob)
        self.ffn = FeedForward(
            dim_model,
            dim_ffn,
            hidden_dropout_prob,
        )

    def forward(self, x: Tensor, rotary_pos_enc: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tensor:
        """ rel_pos_emb is fixed for all layer in same forward pass time """
        ln_x = self.layer_norm1(x)
        residual_x = self.hidden_dropout(self.self_attention(ln_x, rotary_pos_enc, padding_mask, attention_mask)) + x
        ln_x = self.layer_norm2(residual_x)
        fx = self.ffn(ln_x) + residual_x
        return fx


class RoformerEncoder(nn.Module, AbstractModel):
    """ In this class,
        1) encode input sequence,
        2) make absolute position embedding,
        3) matrix sum with word embedding, absolute position embedding
        4) stack num_layers BERTEncoderLayer

    Output have ONLY result of pure self-attention
    Args:
        max_seq: maximum sequence length, named "max_position_embedding" in official repo, default 512, in official paper, this value is called 'k'
        num_layers: number of EncoderLayer, default 6 for base model

    References:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/pdf/1810.04805.pdf
        https://arxiv.org/abs/2006.16236
        https://arxiv.org/abs/2104.09864  # RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py
        https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
    """
    def __init__(
            self,
            cfg: CFG,
            max_seq: int = 512,
            num_layers: int = 12,
            dim_model: int = 768,
            num_attention_heads: int = 12,
            dim_ffn: int = 3072,
            kernel: str = 'elu',
            layer_norm_eps: float = 0.02,
            attention_dropout_prob: float = 0.1,
            hidden_dropout_prob: float = 0.1,
            gradient_checkpointing: bool = False
    ) -> None:
        super(RoformerEncoder, self).__init__()
        self.cfg = cfg
        self.max_seq = max_seq
        self.num_layers = num_layers
        self.dim_model = dim_model
        self.num_attention_heads = num_attention_heads
        self.dim_ffn = dim_ffn
        self.hidden_dropout = nn.Dropout(p=hidden_dropout_prob)  # dropout is not learnable
        self.layer = nn.ModuleList(
            [RoformerEncoderLayer(dim_model, num_attention_heads, dim_ffn, kernel, layer_norm_eps, attention_dropout_prob, hidden_dropout_prob) for _ in range(self.num_layers)]
        )
        self.layer_norm = nn.LayerNorm(dim_model, eps=layer_norm_eps)  # for final-Encoder output
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, inputs: Tensor, rotary_pos_enc: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: embedding from input sequence
            rotary_pos_enc: rotary position encoding, shape (batch_size, seq_len, dim_model, dim_model)
            padding_mask: mask for Encoder padded token for speeding up to calculate attention score or MLM
            attention_mask: mask for CLM
        """
        layer_output = []
        x = inputs
        for layer in self.layer:
            if self.gradient_checkpointing and self.cfg.train:
                x = self._gradient_checkpointing_func(
                    layer.__call__,
                    x,
                    rotary_pos_enc,
                    padding_mask,
                    attention_mask
                )
            else:
                x = layer(
                    x,
                    rotary_pos_enc,
                    padding_mask,
                    attention_mask
                )
            layer_output.append(x)
        last_hidden_state = self.layer_norm(x)  # because of applying pre-layer norm
        hidden_states = torch.stack(layer_output, dim=0).to(x.device)  # shape: [num_layers, BS, SEQ_LEN, DIM_Model]
        return last_hidden_state, hidden_states


class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """ This module produces sinusoidal positional embeddings of any length
    Original Source code from Huggingface's RoFormer model, which is the most optimized way to create positional embedding

    Args:
        max_seq: max sequence length of model
        dim_head: dimension of each attention head's hidden states

    References:
        https://arxiv.org/abs/2104.09864  # RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py#L323
    """

    def __init__(self, max_seq: int, dim_head: int) -> None:
        super().__init__(max_seq, dim_head)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, seq_len: int, past_key_values_length: int = 0) -> Tensor:
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)


class Embedding(nn.Module):
    """ Class module for Roformer Embedding, word embedding & rotary positional encoding
    This module has option => whether or not to use ALBERT Style Factorized Embedding

    Very Un-Optimized way to apply rotary position encoding to word embedding
    Notes:
         ASAP, we will implement more optimized way to apply rotary position encoding to word embedding

    This Module set & initialize 3 Embedding Layers:
        1) Word Embedding
        2) Rotary Positional Encoding

    Args:
        cfg: configuration.py

    References:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/pdf/1810.04805.pdf
        https://arxiv.org/abs/2006.16236
        https://arxiv.org/abs/2104.09864  # RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py
        https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
    """
    def __init__(self, cfg: CFG) -> None:
        super(Embedding, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.max_seq = cfg.max_seq
        self.dim_model = cfg.dim_model
        self.word_embedding = nn.Embedding(len(cfg.tokenizer), cfg.dim_model)
        self.layer_norm1 = nn.LayerNorm(cfg.dim_model, eps=cfg.layer_norm_eps)  # for word embedding
        self.hidden_dropout = nn.Dropout(p=cfg.hidden_dropout_prob)
        # self.rotary_pos_encoding = self.create_rotation_matrix
        self.rotary_pos_encoding = RoFormerSinusoidalPositionalEmbedding(
            cfg.max_seq,
            cfg.dim_model // cfg.num_attention_heads
        )

        # ALBERT Style Factorized Embedding
        if self.cfg.is_mf_embedding:
            self.word_embedding = nn.Embedding(len(cfg.tokenizer), int(cfg.dim_model/6))
            self.projector = nn.Linear(int(cfg.dim_model/6), cfg.dim_model)  # project to original hidden dim

    @torch.no_grad()
    def create_rotation_matrix(self, seq_len: int) -> Tensor:
        """ Create a batch of rotation matrices from the given thetas.
        This function must be wrapped with torch.no_grad(), because it's not learnable parameters

        1) Create m*theta matrix (seq_len, dim_model): thetas
            - m: position index
            - theta: positional encoding value from static function (10000**(-2 * (i_arr - 1) / self.dim_model))

        2) Create R matrix (seq_len, dim_model, dim_model): R
            - example:
                [cos m*theta1, -sin m*theta1, 0, 0]
                [sin m*theta1, cos m*theta1, 0, 0]
                [0, 0, cos m*theta2, -sin m*theta2]
                [0, 0, sin m*theta2, cos m*theta2]

        Args:
            seq_len: max sequence length in batch

        Returns:
            Tensor: A tensor of shape (batch_size, seq_len, d, d) containing the rotation matrices.
        """
        i_arr = torch.arange(1, int(self.dim_model / 2) + 1).repeat_interleave(2).to(self.cfg.device)  # for rotary position embedding
        theta = 10000**(-2 * (i_arr - 1) / self.dim_model)  # for rotary position embedding
        scaler = torch.arange(1, seq_len + 1, device=self.cfg.device, dtype=torch.float).unsqueeze(1).repeat(1, self.dim_model).reshape(seq_len, self.dim_model)
        thetas = torch.mul(scaler, theta)

        R = torch.eye(self.dim_model, device=thetas.device).repeat(seq_len, 1, 1)
        for i in range(0, self.dim_model, 2):
            cos_t = torch.cos(thetas[:, i]).unsqueeze(-1)
            sin_t = torch.sin(thetas[:, i]).unsqueeze(-1)

            R[:, i, i] = cos_t.squeeze(-1)
            R[:, i + 1, i + 1] = cos_t.squeeze(-1)
            R[:, i, i + 1] = -sin_t.squeeze(-1)
            R[:, i + 1, i] = sin_t.squeeze(-1)

        return R

    def forward(self, inputs: Tensor) -> Tuple[nn.Embedding, Tensor]:
        if self.cfg.is_mf_embedding:
            word_embeddings = self.hidden_dropout(
                self.layer_norm1(self.projector(self.word_embedding(inputs)))
            )
        else:
            word_embeddings = self.hidden_dropout(
                self.layer_norm1(self.word_embedding(inputs))
            )
        rotary_pos_enc = self.rotary_pos_encoding(inputs.shape[1])
        return word_embeddings, rotary_pos_enc


class Roformer(nn.Module, AbstractModel):
    """ Main class for Roformer, having all of sub-blocks & modules such as Disentangled Self-attention, Encoder
    Init Scale of Roformer Hyper-Parameters, Embedding Layer, Encoder Blocks
    And then make 2-types of Embedding Layer, Word Embedding, Absolute Position Embedding

    This module has only Encoder Block, not Decoder Block

    Args:
        cfg: configuration.CFG
        num_layers: number of EncoderLayer, default 12 for base model
        this value must be init by user for objective task
        if you select electra, you should set num_layers twice (generator, discriminator)

    Var:
        vocab_size: size of vocab in RoFormer's Native Tokenizer
        max_seq: maximum sequence length
        max_rel_pos: max_seq x2 for build relative position embedding
        num_layers: number of Disentangled-Encoder layers
        num_attention_heads: number of attention heads
        dim_model: dimension of model
        num_attention_heads: number of heads in multi-head attention
        dim_ffn: dimension of feed-forward network, same as intermediate size in official repo
        kernel: kernel function for attention head
        hidden_dropout_prob: dropout rate for embedding, hidden layer
        attention_probs_dropout_prob: dropout rate for attention

    Maths:
        Q = RdΘ,m φ(q m)
        K = RdΘ,n ϕ(kn)
        A = normalize(Φ(Q).mm(Φ(K).t())).mm(V)  (Linear Attention, otherwise are same as original Transformer, BERT)

    References:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/pdf/1810.04805.pdf
        https://arxiv.org/abs/2006.16236
        https://arxiv.org/abs/2104.09864  # RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py
        https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
    """
    def __init__(self, cfg: CFG, num_layers: int = 12) -> None:
        super(Roformer, self).__init__()
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.max_seq = cfg.max_seq
        self.num_layers = num_layers
        self.num_attention_heads = cfg.num_attention_heads
        self.dim_model = cfg.dim_model
        self.dim_ffn = cfg.dim_ffn
        self.kernel = cfg.kernel
        self.layer_norm_eps = cfg.layer_norm_eps
        self.hidden_dropout_prob = cfg.hidden_dropout_prob
        self.attention_dropout_prob = cfg.attention_probs_dropout_prob
        self.gradient_checkpointing = cfg.gradient_checkpoint

        # Init Embedding Layer
        self.embeddings = Embedding(cfg)

        # Init Encoder Blocks & Modules
        self.encoder = RoformerEncoder(
            self.cfg,
            self.max_seq,
            self.num_layers,
            self.dim_model,
            self.num_attention_heads,
            self.dim_ffn,
            self.kernel,
            self.layer_norm_eps,
            self.attention_dropout_prob,
            self.hidden_dropout_prob,
            self.gradient_checkpointing
        )

    def forward(self, inputs: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: input sequence, shape (batch_size, sequence)
            padding_mask: padding mask for MLM or padding token
            attention_mask: attention mask for CLM, default None
        """
        assert inputs.ndim == 2, f'Expected (batch, sequence) got {inputs.shape}'
        word_embeddings, rotary_pos_enc = self.embeddings(inputs)
        last_hidden_state, hidden_states = self.encoder(
            word_embeddings,
            rotary_pos_enc,
            padding_mask,
            attention_mask
        )
        return last_hidden_state, hidden_states
