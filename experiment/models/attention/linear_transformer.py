import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment.models.abstract_model import AbstractModel
from torch import Tensor
from typing import Tuple, List
from einops.layers.torch import Rearrange
from configuration import CFG


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
    dim_head: int = 64,
    kernel: str = 'elu',
    eps: float = 1e-6,
    attention_dropout: nn.Dropout = None,
    padding_mask: Tensor = None,
    attention_mask: Tensor = None,
) -> Tensor:
    """ DocString will be updated ASAP
    Linear attention with Masking for padding mask

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

    Reference:
        https://arxiv.org/abs/2006.16236
        https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
    """
    projected_q, projected_k = kernel_fn(q, kernel), kernel_fn(k, kernel)
    if padding_mask is not None:  # applying padding mask, calculating normalizer
        projected_q[padding_mask == 1], projected_k[padding_mask == 1], v[padding_mask == 1] = 0, 0, 0

    kv = torch.matmul(v, projected_k.permute(0, 2, 1).contiguous())
    qkv = torch.matmul(projected_q, kv)

    normalizer = projected_k.sum(dim=1).unsqueeze(1).expand(-1, dim_head, -1).permute(0, 2, 1).contiguous()
    z = 1 / torch.clamp(torch.matmul(projected_q, normalizer), min=eps)
    attention_matrix = torch.mul(qkv, z)

    # attention dropout
    if attention_dropout is not None:
        attention_matrix = attention_dropout(
            attention_matrix
        )

    return attention_matrix


class AttentionHead(nn.Module):
    """ In this class, we implement workflow of single attention head in Linear Transformer
    This class has same role as Module "BertAttention" in official Repo (bert.py)

    Args:
        dim_model: dimension of model's latent vector space, default 1024 from official paper
        dim_head: dimension of each attention head, default 64 from official paper (1024 / 16)
        kernel: kernel function for attention head, default 'elu' from official paper
        attention_dropout_prob: dropout rate for attention matrix, default 0.1 from official paper

    Math:
        A = normalize(Φ(Q).mm(Φ(K).t())).mm(V)

    Reference:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/pdf/1810.04805.pdf
        https://arxiv.org/abs/2006.16236
        https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
    """
    def __init__(
            self,
            dim_model: int = 1024,
            dim_head: int = 64,
            kernel: str = 'elu',
            attention_dropout_prob: float = 0.1
    ) -> None:
        super(AttentionHead, self).__init__()
        self.dim_model = dim_model
        self.dim_head = dim_head  # 1024 / 16 = 64
        self.kernel = kernel
        self.eps = 1e-6
        self.attention_dropout = nn.Dropout(p=attention_dropout_prob)
        self.fc_q = nn.Linear(self.dim_model, self.dim_head)
        self.fc_k = nn.Linear(self.dim_model, self.dim_head)
        self.fc_v = nn.Linear(self.dim_model, self.dim_head)

    def forward(self, x: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tensor:
        q, k, v = self.fc_q(x), self.fc_k(x), self.fc_v(x)
        attention_matrix = linear_attention(
            q,
            k,
            v,
            self.dim_head,
            self.kernel,
            self.eps,
            self.attention_dropout,
            padding_mask,
            attention_mask
        )
        return attention_matrix


class MultiHeadAttention(nn.Module):
    """ In this class, we implement workflow of Multi-Head Self-attention for Linear Transformers
    This class has same role as Module "BertAttention" in official Repo (bert.py)
    In official repo, they use post-layer norm, but we use pre-layer norm which is more stable & efficient for training

    Args:
        dim_model: dimension of model's latent vector space, default 1024 from official paper
        num_attention_heads: number of heads in MHSA, default 16 from official paper for Transformer
        dim_head: dimension of each attention head, default 64 from official paper (1024 / 16)
        kernel: kernel function for attention head, default 'elu' from official paper
        attention_dropout_prob: dropout rate, default 0.1

    Math:
        A = softmax(attention Matrix/sqrt(3*D_h)), SA(z) = Av

    Reference:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/pdf/1810.04805.pdf
        https://arxiv.org/abs/2006.16236
        https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
    """
    def __init__(
            self,
            dim_model: int = 1024,
            num_attention_heads: int = 16,
            dim_head: int = 64,
            kernel: str = 'elu',
            attention_dropout_prob: float = 0.1
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.dim_model = dim_model
        self.num_attention_heads = num_attention_heads
        self.dim_head = dim_head
        self.kernel = kernel
        self.attention_dropout_prob = attention_dropout_prob
        self.attention_heads = nn.ModuleList(
            [AttentionHead(self.dim_model, self.dim_head, self.kernel, self.attention_dropout_prob) for _ in range(self.num_attention_heads)]
        )
        self.fc_concat = nn.Linear(self.dim_model, self.dim_model)

    def forward(self, x: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tensor:
        """ x is already passed nn.Layernorm """
        assert x.ndim == 3, f'Expected (batch, seq, hidden) got {x.shape}'
        attention_output = self.fc_concat(
            torch.cat([head(x, padding_mask, attention_mask) for head in self.attention_heads], dim=-1)
        )
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


class LinearTransformerEncoderLayer(nn.Module):
    """ Class for encoder model module in Linear Transformer
    In this class, we stack each encoder_model module (Multi-Head attention, Residual-Connection, LayerNorm, FFN)
    This class has same role as Module "BertEncoder" in official Repo (bert.py)

    References:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/pdf/1810.04805.pdf
        https://arxiv.org/abs/2006.16236
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
        super(LinearTransformerEncoderLayer, self).__init__()
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

    def forward(self, x: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tensor:
        """ rel_pos_emb is fixed for all layer in same forward pass time """
        ln_x = self.layer_norm1(x)
        residual_x = self.hidden_dropout(
            self.self_attention(ln_x, padding_mask, attention_mask)
        ) + x

        ln_x = self.layer_norm2(residual_x)
        fx = self.ffn(ln_x) + residual_x
        return fx


class LinearTransformerEncoder(nn.Module, AbstractModel):
    """ In this class,
        1) encode input sequence,
        2) make absolute position embedding,
        3) matrix sum with word embedding, absolute position embedding
        4) stack num_layers BERTEncoderLayer
    Output have ONLY result of pure self-attention

    Args:
        max_seq: maximum sequence length, named "max_position_embedding" in official repo, default 512, in official paper, this value is called 'k'
        num_layers: number of EncoderLayer, default 6 for base model
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
        super(LinearTransformerEncoder, self).__init__()
        self.cfg = cfg
        self.max_seq = max_seq
        self.num_layers = num_layers
        self.dim_model = dim_model
        self.num_attention_heads = num_attention_heads
        self.dim_ffn = dim_ffn
        self.hidden_dropout = nn.Dropout(p=hidden_dropout_prob)  # dropout is not learnable
        self.layer = nn.ModuleList(
            [LinearTransformerEncoderLayer(dim_model, num_attention_heads, dim_ffn, kernel, layer_norm_eps, attention_dropout_prob, hidden_dropout_prob) for _ in range(self.num_layers)]
        )
        self.layer_norm = nn.LayerNorm(dim_model, eps=layer_norm_eps)  # for final-Encoder output
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, inputs: Tensor, abs_pos_emb: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: embedding from input sequence
            abs_pos_emb: absolute position embedding
            padding_mask: mask for Encoder padded token for speeding up to calculate attention score or MLM
            attention_mask: mask for CLM
        """
        layer_output = []
        x = inputs + abs_pos_emb  # pass to layer
        for layer in self.layer:
            if self.gradient_checkpointing and self.cfg.train:
                x = self._gradient_checkpointing_func(
                    layer.__call__,
                    x,
                    padding_mask,
                    attention_mask
                )
            else:
                x = layer(
                    x,
                    padding_mask,
                    attention_mask
                )
            layer_output.append(x)
        last_hidden_state = self.layer_norm(x)  # because of applying pre-layer norm
        hidden_states = torch.stack(layer_output, dim=0).to(x.device)  # shape: [num_layers, BS, SEQ_LEN, DIM_Model]
        return last_hidden_state, hidden_states


class Embedding(nn.Module):
    """ Class module for RoFormer Embedding, word embedding & rotary positional encoding
    This module has option => whether or not to use ALBERT Style Factorized Embedding
    This Module set & initialize 3 Embedding Layers:
        1) Word Embedding 2) Rotary Positional Encoding

    Args:
        cfg: configuration.py

    Notes:
        Rotary Positional Encoding added at bottom layers
    """
    def __init__(self, cfg: CFG) -> None:
        super(Embedding, self).__init__()
        self.cfg = cfg
        self.max_seq = cfg.max_seq
        self.word_embedding = nn.Embedding(len(cfg.tokenizer), cfg.dim_model)
        self.abs_pos_emb = nn.Embedding(cfg.max_seq, cfg.dim_model)  # Absolute Position Embedding for EMD Layer
        self.layer_norm1 = nn.LayerNorm(cfg.dim_model, eps=cfg.layer_norm_eps)  # for word embedding
        self.hidden_dropout = nn.Dropout(p=cfg.hidden_dropout_prob)

        # ALBERT Style Factorized Embedding
        if self.cfg.is_mf_embedding:
            self.word_embedding = nn.Embedding(len(cfg.tokenizer), int(cfg.dim_model/6))
            self.projector = nn.Linear(int(cfg.dim_model/6), cfg.dim_model)  # project to original hidden dim

    def forward(self, inputs: Tensor) -> Tuple[nn.Embedding, nn.Embedding]:
        if self.cfg.is_mf_embedding:
            word_embeddings = self.hidden_dropout(
                self.layer_norm1(self.projector(self.word_embedding(inputs)))
            )
        else:
            word_embeddings = self.hidden_dropout(
                self.layer_norm1(self.word_embedding(inputs))
            )
        rotary_pos_encoding = self.abs_pos_emb(
            torch.arange(inputs.shape[1], device="cuda").repeat(inputs.shape[0]).view(inputs.shape[0], -1)
        )
        return word_embeddings, rotary_pos_encoding


class LinearTransformer(nn.Module, AbstractModel):
    """
    Main class for LinearTransformer, having all of sub-blocks & modules such as Disentangled Self-attention, Encoder
    Init Scale of LinearTransformer Hyper-Parameters, Embedding Layer, Encoder Blocks
    And then make 3-types of Embedding Layer, Word Embedding, Absolute Position Embedding, Relative Position Embedding

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
        A = normalize(Φ(Q).mm(Φ(K).t())).mm(V)  (Linear Attention, otherwise are same as original Transformer, BERT)

    References:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/pdf/1810.04805.pdf
        https://arxiv.org/abs/2006.16236
        https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
    """
    def __init__(self, cfg: CFG, num_layers: int = 12) -> None:
        super(LinearTransformer, self).__init__()
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
        self.encoder = LinearTransformerEncoder(
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
        word_embeddings, abs_pos_emb = self.embeddings(inputs)
        last_hidden_state, hidden_states = self.encoder(
            word_embeddings,
            abs_pos_emb,
            padding_mask,
            attention_mask
        )
        return last_hidden_state, hidden_states
