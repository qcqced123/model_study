"""py module for implementation of Titans from Google Research,
there are four variant of Titans architecture

1) MAC Titans (Memory as a Context)
2) MAG Titans (Memory as a Gate)
3) MAL Titans (Memory as a Layer)
4) LMM Titans (No attention)
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple, List
from einops.layers.torch import Rearrange
from transformers import AutoConfig, AutoTokenizer
from experiment.losses.loss import NeuralMemoryLoss
from experiment.activation.activation import SwiGLU
from experiment.models.attention.moe import SparseMoELayer
from experiment.models.abstract_model import AbstractModel


def apply_rotary_position_embeddings(sinusoidal_pos: Tensor, query_layer: Tensor, key_layer: Tensor, value_layer: Tensor = None):
    """ Apply rotary position encoding to query, key layer
    Original Source code from Huggingface's RoFormer model, which is the most optimized way to create positional embedding

    You can find mathematical proof in official paper's Appendix

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


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dot_scale: Tensor,
    attention_dropout: nn.Dropout,
    attention_mask: Tensor = None,
) -> Tensor:
    """ Scaled Dot-Product attention with Masking for attention mask and padding mask

    Args:
        q: query matrix, shape (batch_size, seq_len, dim_head)
        k: key matrix, shape (batch_size, seq_len, dim_head)
        v: value matrix, shape (batch_size, seq_len, dim_head)
        dot_scale: scale factor for Q•K^T result
        attention_dropout: dropout for attention matrix, default rate is 0.1 from official paper
        attention_mask: mask for attention matrix for CLM, this tensor is already combined with padding_mask

    Math:
        A = softmax(q•k^t/sqrt(D_h)), SA(z) = Av

    Reference:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/abs/2005.14165
        https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
        https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    """
    BS, NUM_HEADS, SEQ_LEN, DIM_HEADS = q.shape
    attention_matrix = torch.matmul(q, k.permute(0, 2, 3, 1).contiguous()) / dot_scale

    # apply the attention mask for restricting the using future information
    # for broadcasting to attention matrix, shape: (BS, 1, SEQ_LEN, SEQ_LEN)
    if attention_mask is not None:
        attention_mask = attention_mask.unsqueeze(1)
        attention_matrix = attention_matrix.masked_fill(attention_mask == 1, float('-inf'))

    attention_dist = attention_dropout(
        F.softmax(attention_matrix, dim=-1)
    )
    attention_matrix = torch.matmul(attention_dist, v).permute(0, 2, 1, 3).reshape(-1, SEQ_LEN, NUM_HEADS*DIM_HEADS).contiguous()
    return attention_matrix


def separable_convolution_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
):
    """ func of "1D separable depth-wise convolution", replacing the "global full attention", firstly suggested by "mobileNet"

    Args:
        q: query matrix, shape (batch_size, seq_len, dim_head)
        k: key matrix, shape (batch_size, seq_len, dim_head)
        v: value matrix, shape (batch_size, seq_len, dim_head)

    reference:
        - https://arxiv.org/pdf/1704.04861
        - https://arxiv.org/pdf/2111.00396
    """

    return


def sliding_window_attention():
    """ sliding window(local, block-sparse, ...) attention func for encoding the "short-term memory" for MAG Titans
    """
    return


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        arch: str,
        dim_head: int = 64,
        dim_model: int = 1024,
        num_attention_heads: int = 16,
        attention_dropout_prob: float = 0.1
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.dim_head = dim_head
        self.dim_model = dim_model
        self.num_attention_heads = num_attention_heads

        # init necessary module
        self.fc_q = nn.Linear(self.dim_model, self.dim_model)
        self.fc_k = nn.Linear(self.dim_model, self.dim_model)
        self.fc_v = nn.Linear(self.dim_model, self.dim_model)
        self.fc_concat = nn.Linear(self.dim_model, self.dim_model)  # same as W_O in original paper
        self.apply_rope = apply_rotary_position_embeddings
        self.attention = scaled_dot_product_attention if arch == "MAC" else sliding_window_attention

        self.attention_dropout = nn.Dropout(p=attention_dropout_prob)
        self.dot_scale = torch.sqrt(torch.tensor(self.dim_head, dtype=torch.float32))


    def forward(self, x: Tensor, rotary_pos_enc: Tensor, attention_mask: Tensor = None) -> Tensor:
        """ x is already passed nn.Layernorm, already multiplied with rotary position encoding """
        assert x.ndim == 3, f'Expected (batch, seq, hidden) got {x.shape}'

        # size: bs, seq, nums head, dim head, linear projection
        # project the word embedding vector to each latent space (query, key, value)
        q = self.fc_q(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        k = self.fc_k(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        v = self.fc_v(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        # multiply word embedding, rotary position encoding for encoding word embedding vector to absolute, relative position embedding
        rotary_q, rotary_k = self.apply_rope(rotary_pos_enc, q, k)

        # do self-attention with multi-heads
        attention_matrix = self.attention(
            rotary_q,
            rotary_k,
            v,
            self.dot_scale,
            self.attention_dropout,
            attention_mask
        )

        # mixture of latent vector from multi-heads
        attention_output = self.fc_concat(attention_matrix)
        return attention_output


class ShortTermMemoryLayer(nn.Module):
    def __init__(
        self,
        arch: str = "MAC",
        dim_ffn: int = 3072,
        dim_model: int = 1024,
        num_attention_heads: int = 16,
        layer_norm_eps: float = 0.02,
        hidden_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
    ) -> None:
        super(ShortTermMemoryLayer, self).__init__()
        self.arch = arch
        self.ffn = dim_ffn
        self.dim_model = dim_model
        self.num_attention_heads = num_attention_heads
        self.attention_dropout_prob = attention_dropout_prob
        self.dim_heads = int(dim_model // num_attention_heads)

        # init the necessary module of encoder/decoder
        self.attention_heads = MultiHeadAttention(
            arch=self.arch,
            dim_head=self.dim_heads,
            dim_model=self.dim_model,
            num_attention_heads=self.num_attention_heads,
            attention_dropout_prob=self.attention_dropout_prob
        )
        self.ffn = SwiGLU(
            dim_model=self.dim_model,
            dim_ffn=self.dim_ffn
        )
        self.pre_ln = nn.RMSNorm(
            normalized_shape=dim_model,
            eps=layer_norm_eps
        )
        self.post_ln = nn.RMSNorm(
            normalized_shape=dim_model,
            eps=layer_norm_eps
        )
        self.dropout = nn.Dropout(p=hidden_dropout_prob)

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        ln_x = self.pre_ln(x)
        residual_x = self.dropout(self.attention_heads(ln_x, attention_mask)) + x
        ln_x = self.post_ln(residual_x)
        hx = self.ffn(ln_x) + residual_x
        return hx


class ShortTermMemory(nn.Module, AbstractModel):
    """ module that is responsible to store/remember/encode to "short-term memory" by using "limited window attention",
    named in original paper as "Core".

    this module has same role as transformer-based model's encoder/decoder module.
    in test/inference time, this module do the "in-context learning", same as "few-shot learning" as well.
    so on, this module is fixed in inference time, not updating weight parameters
    """
    def __init__(
        self,
        cfg,
        max_seq: int = 512,
        num_layers: int = 12,
        num_heads: int = 16,
        dim_model: int = 1024,
        dim_ffn: int = 3072,
        layer_norm_eps: float = 0.02,
        attention_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        gradient_checkpointing: bool = False
    ) -> None:
        super(ShortTermMemory, self).__init__()
        self.cfg = cfg
        self.max_seq = max_seq
        self.num_heads = num_heads
        self.dim_ffn = dim_ffn
        self.dim_model = dim_model
        self.num_layers = num_layers
        self.gradient_checkpointing = gradient_checkpointing

        # init the necessary module of short-term memory
        self.layer = nn.ModuleList(
            [ShortTermMemoryLayer(cfg.arch, self.dim_ffn, self.dim_model, self.num_heads, layer_norm_eps, hidden_dropout_prob, attention_dropout_prob) for _ in range(self.num_layers)]
        )
        self.layer_norm = nn.RMSNorm(
            normalized_shape=dim_model,
            eps=layer_norm_eps
        )
        self.dropout = nn.Dropout(p=hidden_dropout_prob)

    def forward(self, inputs: Tensor, rotary_pos_enc: Tensor, attention_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: embedding from input sequence
            rotary_pos_enc: rotary position encoding, shape (batch_size, seq_len, dim_model, dim_model)
            attention_mask: mask for casual language modeling
        """
        x = inputs
        layer_output = []
        for layer in self.layer:
            if self.gradient_checkpointing and self.cfg.train:
                x = self._gradient_checkpointing_func(
                    layer.__call__,
                    x,
                    rotary_pos_enc,
                    attention_mask
                )
            else:
                x = layer(
                    x,
                    rotary_pos_enc,
                    attention_mask
                )
            layer_output.append(x)

        # post layer-norm for output hidden state vector, stacking the all of intermediate layers
        last_hidden_state = self.layer_norm(x)  # because of applying pre-layer norm
        hidden_states = torch.stack(layer_output, dim=0).to(x.device)  # shape: [num_layers, BS, SEQ_LEN, DIM_Model]
        return last_hidden_state, hidden_states


class LongTermMemory(nn.Module):
    """ module that is responsible to store/remember long past, named in original paper as "Contextual Memory", "Neural Memory"
    in test/inference time, this module will be updated by test/inference data, similar as "RAG" document db as well.

    this module measure the "surprise" of an input with the gradient of the neural network
    with respect to the input in associative memory loss.

    the more "surprise", the more memorable data/abstraction/memory.

    objective function:
        1) weight update: l(M_{t-1}; x_t) = || M_{t-1}(k_t) - v_t ||_2^2
        2) retrieve: y_t = M^*(q_t)

    Args:
        dim_ffn (int):
        dim_model (int):
        num_layers (int):
        hidden_dropout_prob (float):
        gradient_checkpointing (bool):
    """
    def __init__(
        self,
        dim_ffn: int,
        dim_model: int = 1024,
        num_layers: int = 4,
        layer_norm_eps: float = 0.02,
        hidden_dropout_prob: float = 0.1,
        gradient_checkpointing: bool = False
    ) -> None:
        super(LongTermMemory, self).__init__()
        self.dim_ffn = dim_ffn
        self.dim_model = dim_model
        self.num_layers = num_layers
        self.project_q = nn.Linear(self.dim_model, self.dim_model)  # for retrieve x
        self.project_k = nn.Linear(self.dim_model, self.dim_model)  # for online updating
        self.project_v = nn.Linear(self.dim_model, self.dim_model)  # for online updating
        self.layer = nn.Sequential(
            nn.Linear(self.dim_model, self.dim_ffn),
            nn.SiLU(),
            nn.Dropout(p=hidden_dropout_prob),
            nn.Linear(self.dim_ffn, self.dim_model),
            nn.Dropout(p=hidden_dropout_prob),
        )
        self.neural_memory = nn.ModuleList(
            [self.layer for _ in range(self.num_layers)]
        )
        self.gradient_checkpointing = gradient_checkpointing
        self.layer_norm = nn.RMSNorm(
            normalized_shape=dim_model,
            eps=layer_norm_eps
        )
        self.criterion = NeuralMemoryLoss()  # for online meta-model

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Args:
            x (Tensor): input x of the current time "t"
        """
        qx, kx, vx = self.project_q(x), self.project_k(x), self.project_v(x)
        for layer in self.neural_memory:
            qx, kx = layer(qx), layer(kx)

        cnt_y = self.layer_norm(qx)
        cnt_loss = self.criterion(k=kx, v=vx)
        return cnt_y, cnt_loss


class PersistentMemory(nn.Module):
    """ module of learnable but date-independent parameters that encodes the knowledge about a task,
    role as meta/common-sense knowledge in human-being, named in original paper as "Persistent Memory".

    this module have the similar architecture as feedforward network of transformer-base model, replacing the ReLU/GELU ... in fully connected layer to Softmax.
    in test/inference time, this module will be fixed.

    we think this module has the similar role as prompt encoder in "prompt learning" method or learnable initial token from "streamingLLM"

    Args:
        dim_model (int):
        hidden_dropout_prob (float):
    """

    def __init__(self, dim_model: int, hidden_dropout_prob: float = 0.1) -> None:
        super(PersistentMemory, self).__init__()
        self.projector_k = nn.Linear(dim_model, dim_model)
        self.projector_v = nn.Linear(dim_model, dim_model)
        self.activation_func = nn.Softmax(dim=-1)
        self.ffn_dropout = nn.Dropout(p=hidden_dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x (Tensor): input x in current time "t"
        """
        kx = self.projector_k(x)
        hx = self.ffn_dropout(self.activation_func(kx))
        px = self.ffn_dropout(self.projector_v(hx))
        return px


class RotarySinusoidalPositionalEmbedding(nn.Embedding):
    """ This module produces rotary sinusoidal positional embeddings of any length
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
        """ identical to the XLM create_sinusoidal_embeddings except features are not interleaved.
        The cos features are in the 2nd half of the vector. [dim // 2:].
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
    """ module of word embedding, RoPE(Rotary Position) embedding in pytorch implementation
    Args:
        cfg: configuration.py

    References:
        https://arxiv.org/abs/2104.09864  # RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py
    """
    def __init__(self, cfg) -> None:
        super(Embedding, self).__init__()
        self.cfg = cfg
        self.max_seq = cfg.max_seq
        self.dim_model = cfg.dim_model
        self.batch_size = cfg.batch_size
        self.word_embedding = nn.Embedding(
            len(cfg.tokenizer),
            cfg.dim_model
        )
        self.rotary_pos_encoding = RotarySinusoidalPositionalEmbedding(
            cfg.max_seq,
            cfg.dim_model // cfg.num_attention_heads
        )
        self.layer_norm = nn.RMSNorm(
            normalized_shape=cfg.dim_model,
            eps=cfg.layer_norm_eps
        )
        self.hidden_dropout = nn.Dropout(
            p=cfg.hidden_dropout_prob
        )

    def pre_forward(self, inputs: Tensor) -> Tensor:
        word_embeddings = self.hidden_dropout(
            self.layer_norm(self.word_embedding(inputs))
        )
        return word_embeddings

    def forward(self, x: Tensor) -> Tensor:
        rotary_pos_enc = self.rotary_pos_encoding(x.shape[1])
        return rotary_pos_enc


class MACTitans(nn.Module, AbstractModel):
    """ interface module of MAC(Memory as a Context Architecture) Titans

    Design Point:
        - RoPE embedding
        - global full attention
        - pre layer-normalization (rms layer-normalization)
        - non-linear activation func: silu
        - mlp layer: gated linear unit with silu for short-term memory, feedforward for long-term, persistent memory

    Args:
        cfg: configuration module of initializing the current model

    Reference:
        https://arxiv.org/pdf/2501.00663
    """
    def __init__(self, cfg):
        super(MACTitans, self).__init__()
        self.cfg = cfg
        self.max_seq = cfg.max_seq  # maximum size of context window
        self.vocab_size = cfg.vocab_size  # size of maximum vocabulary
        self.num_layers = cfg.num_layers
        self.num_attention_heads = cfg.num_attention_heads

        self.dim_ffn = cfg.dim_ffn
        self.dim_model = cfg.dim_model
        self.layer_norm_eps = cfg.layer_norm_eps  # epsilon of layer-norm for numerical stability
        self.hidden_dropout_prob = cfg.hidden_dropout_prob  # probs of dropout on model, except only self-attention
        self.attention_dropout_prob = cfg.attention_probs_dropout_prob  # probs of dropout at self-attention layer
        self.gradient_checkpointing = cfg.gradient_checkpoint  # flag variable of gradient checkpointing

        # init necessary module
        self.embeddings = Embedding(cfg)
        self.short_term_memory = ShortTermMemory(
            cfg=cfg,
            max_seq=self.max_seq,
            num_layers=self.num_layers,
            num_heads=self.num_attention_heads,
            dim_model=self.dim_model,
            dim_ffn=self.dim_ffn,
            layer_norm_eps=self.layer_norm_eps,
            attention_dropout_prob=self.attention_dropout_prob,
            hidden_dropout_prob=self.hidden_dropout_prob,
            gradient_checkpointing=self.gradient_checkpointing
        )
        self.long_term_memory = LongTermMemory(
            dim_ffn=self.dim_ffn,
            dim_model=self.dim_model,
            num_layers=self.num_layers,
            layer_norm_eps=self.layer_norm_eps,
            hidden_dropout_prob=self.hidden_dropout_prob
        )
        self.persistent_memory = PersistentMemory(
            dim_model=self.dim_model,
            hidden_dropout_prob=self.hidden_dropout_prob
        )

    def forward(self, inputs: Tensor, attention_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: input sequence, shape (batch_size, sequence)
            attention_mask: attention mask for CLM, already combined with padding mask, default None
        """
        assert inputs.ndim == 2, f'Expected (batch, sequence) got {inputs.shape}'
        # retrieve the persistent memory and long-term memory
        word_embeddings = self.embedding.pre_forward(inputs)
        meta_memory = self.persistent_memory(word_embeddings)
        contextual_memory, long_term_loss = self.long_term_memory(
            word_embeddings
        )

        # concatenate the persistent memory, long-term memory, short-term memory
        # get the rotary position embedding vector
        x = torch.concat([meta_memory, contextual_memory, word_embeddings], dim=1)
        rotary_pos_enc = self.embeddings(x)
        short_term_y, hidden_states = self.short_term_memory(
            x,
            rotary_pos_enc,
            attention_mask
        )
        long_term_y = self.long_term_memory(
            short_term_y
        )
        last_hidden_state = torch.mul(short_term_y, long_term_y)
        return last_hidden_state, long_term_loss


class MAGTitans(nn.Module):
    """ interface module of MAG(Memory as a Gate Architecture) Titans
    """
    def __init__(self, cfg):
        super(MAGTitans, self).__init__()

    def forward(self):
        return


class MALTitans(nn.Module):
    """ interface module of MAL(Memory as a Layer Architecture) Titans
    """
    def __init__(self):
        super(MALTitans, self).__init__()

    def forward(self):
        return


class Titans(nn.Module):
    """ interface module of Titans architecture from Google Research, implemented by pytorch
    Args:

    Reference:
        https://arxiv.org/pdf/2501.00663
    """
    def __init__(self):
        super(Titans, self).__init__()

    def forward(self):
        return


if __name__ == '__main__':
    batch_size = 16
    max_seq = 512
    dim_model = 1024
    x = torch.randn(batch_size, max_seq, dim_model)