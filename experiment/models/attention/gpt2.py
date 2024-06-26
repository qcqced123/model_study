import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment.models.abstract_model import AbstractModel
from torch import Tensor
from typing import Tuple, List
from einops.layers.torch import Rearrange
from configuration import CFG


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
        https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
        https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
        https://arxiv.org/abs/2005.14165
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


class MultiHeadAttention(nn.Module):
    """ In this class, we implement workflow of Multi-Head Self-attention for GPT2

    Args:
        dim_model: dimension of model's latent vector space, default 1024 from official paper
        num_attention_heads: number of heads in MHSA, default 16 from official paper for Transformer
        dim_head: dimension of each attention head, default 64 from official paper (1024 / 16)
        attention_dropout_prob: dropout rate, default 0.1

    Math:
        A = softmax(attention Matrix/sqrt(3*D_h)), SA(z) = Av

    Reference:
        https://arxiv.org/abs/1706.03762
        https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
        https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
        https://arxiv.org/abs/2005.14165
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
        self.attention = scaled_dot_product_attention
        self.dot_scale = torch.sqrt(torch.tensor(self.dim_head)).to('cuda')
        self.attention_dropout = nn.Dropout(p=attention_dropout_prob)

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
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
            attention_mask
        )
        attention_output = self.fc_concat(attention_matrix)
        return attention_output


class FeedForward(nn.Module):
    """ Class for Feed-Forward Network module in Transformer Encoder Block, this module for GPT2

    Args:
        dim_model: dimension of model's latent vector space, default 1024
        dim_ffn: dimension of FFN's hidden layer, default 4096 from official paper
        hidden_dropout_prob: dropout rate, default 0.1

    Math:
        FeedForward(x) = FeedForward(LN(x))+x

    Reference:
        https://arxiv.org/abs/1706.03762
        https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
        https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
        https://arxiv.org/abs/2005.14165
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


class GPTDecoderLayer(nn.Module):
    def __init__(
            self,
            dim_model: int = 1024,
            num_attention_heads: int = 16,
            dim_ffn: int = 4096,
            layer_norm_eps: float = 0.02,
            attention_dropout_prob: float = 0.1,
            hidden_dropout_prob: float = 0.1
    ) -> None:
        super(GPTDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            dim_model,
            num_attention_heads,
            int(dim_model / num_attention_heads),
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

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        """ rel_pos_emb is fixed for all layer in same forward pass time """
        ln_x = self.layer_norm1(x)
        residual_x = self.hidden_dropout(
            self.self_attention(ln_x, attention_mask)
        ) + x

        ln_x = self.layer_norm2(residual_x)
        fx = self.ffn(ln_x) + residual_x
        return fx


class GPTDecoder(nn.Module):
    def __init__(
        self,
        cfg: CFG,
        max_seq: int = 512,
        num_layers: int = 12,
        dim_model: int = 768,
        num_attention_heads: int = 12,
        dim_ffn: int = 3072,
        layer_norm_eps: float = 0.02,
        attention_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        gradient_checkpointing: bool = False
    ) -> None:
        super(GPTDecoder, self).__init__()
        self.cfg = cfg
        self.max_seq = max_seq
        self.num_layers = num_layers
        self.dim_model = dim_model
        self.num_attention_heads = num_attention_heads
        self.dim_ffn = dim_ffn
        self.hidden_dropout = nn.Dropout(p=hidden_dropout_prob)  # dropout is not learnable
        self.layer = nn.ModuleList(
            [GPTDecoderLayer(dim_model, num_attention_heads, dim_ffn, layer_norm_eps, attention_dropout_prob, hidden_dropout_prob) for _ in range(self.num_layers)]
        )
        self.layer_norm = nn.LayerNorm(dim_model, eps=layer_norm_eps)  # for final-Encoder output
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, inputs: Tensor, abs_pos_emb: Tensor, attention_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: embedding from input sequence
            abs_pos_emb: absolute position embedding
            attention_mask: mask for CLM
        """
        layer_output = []
        x = inputs + abs_pos_emb  # add absolute position embedding with word embedding
        for layer in self.layer:
            if self.gradient_checkpointing and self.cfg.train:
                x = self._gradient_checkpointing_func(
                    layer.__call__,
                    x,
                    attention_mask
                )
            else:
                x = layer(
                    x,
                    attention_mask
                )
            layer_output.append(x)
        last_hidden_state = self.layer_norm(x)  # because of applying pre-layer norm
        hidden_states = torch.stack(layer_output, dim=0).to(x.device)  # shape: [num_layers, BS, SEQ_LEN, DIM_Model]
        return last_hidden_state, hidden_states


class Embedding(nn.Module):
    """ GPT2 Embedding Module class
    This Module set & initialize 3 Embedding Layers:
        1) Word Embedding 2) Absolute Positional Embedding
    Args:
        cfg: configuration.py
    Notes:
        Absolute Positional Embedding added at bottom layers
    """
    def __init__(self, cfg: CFG) -> None:
        super(Embedding, self).__init__()
        self.cfg = cfg
        self.max_seq = cfg.max_seq
        self.word_embedding = nn.Embedding(len(cfg.tokenizer), cfg.dim_model)
        self.abs_pos_emb = nn.Embedding(cfg.max_seq, cfg.dim_model)
        self.layer_norm1 = nn.LayerNorm(cfg.dim_model, eps=cfg.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(cfg.dim_model, eps=cfg.layer_norm_eps)
        self.hidden_dropout = nn.Dropout(p=cfg.hidden_dropout_prob)

    def forward(self, inputs: Tensor) -> Tuple[nn.Embedding, nn.Embedding]:
        word_embeddings = self.hidden_dropout(
            self.layer_norm1(self.word_embedding(inputs))
        )
        abs_pos_emb = self.hidden_dropout(
            self.layer_norm2(self.abs_pos_emb(torch.arange(inputs.shape[1], device=self.cfg.device).repeat(inputs.shape[0]).view(inputs.shape[0], -1)))
        )
        return word_embeddings, abs_pos_emb


class GPT2(nn.Module, AbstractModel):
    """ Main class for gpt2, which is same model architecture from vanilla transformers decoder, gpt1
    but this version of gpt model has two major change from gpt 1

    1) Pre-Layer Normalization is used instead of Post-Layer Normalization (layernorm module is used before attention and ffn)
    2) Use different Tokenizer: BPE to BBPE (Byte-Pair Encoding to Byte-Level Byte-Pair Encoding)

    original text: x1, x2, x3, x4 ... xt
    masked text: x1, x2, x3, x4, [mask] ... [mask]
                 x1, x2, x3, x4, x5 ... [mask]
    In implementation, we use matrix with attention mask for efficient computation

    References:
        https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
        https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
        https://arxiv.org/abs/2005.14165
    """
    def __init__(self, cfg: CFG, num_layers: int = 12):
        super(GPT2, self).__init__()
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.max_seq = cfg.max_seq
        self.num_layers = num_layers
        self.num_attention_heads = cfg.num_attention_heads
        self.dim_model = cfg.dim_model
        self.dim_ffn = cfg.dim_ffn
        self.layer_norm_eps = cfg.layer_norm_eps
        self.hidden_dropout_prob = cfg.hidden_dropout_prob
        self.attention_dropout_prob = cfg.attention_probs_dropout_prob
        self.gradient_checkpointing = cfg.gradient_checkpoint

        # Init Embedding Layer
        self.embeddings = Embedding(cfg)

        # Init Encoder Blocks & Modules
        self.decoder = GPTDecoder(
            self.cfg,
            self.max_seq,
            self.num_layers,
            self.dim_model,
            self.num_attention_heads,
            self.dim_ffn,
            self.layer_norm_eps,
            self.attention_dropout_prob,
            self.hidden_dropout_prob,
            self.gradient_checkpointing
        )

    def forward(self, inputs: Tensor, attention_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: input sequence, shape (batch_size, sequence)
            attention_mask: attention mask for CLM, already combined with padding mask, default None
        """
        assert inputs.ndim == 2, f'Expected (batch, sequence) got {inputs.shape}'
        word_embeddings, abs_pos_emb = self.embeddings(inputs)
        last_hidden_state, hidden_states = self.decoder(
            word_embeddings,
            abs_pos_emb,
            attention_mask
        )
        return last_hidden_state, hidden_states
