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
    padding_mask: Tensor = None,
    attention_mask: Tensor = None
) -> Tensor:
    """
    Scaled Dot-Product attention with Masking for Decoder
    Args:
        q: query matrix, shape (batch_size, seq_len, dim_head)
        k: key matrix, shape (batch_size, seq_len, dim_head)
        v: value matrix, shape (batch_size, seq_len, dim_head)
        dot_scale: scale factor for Q•K^T result
        attention_dropout: dropout for attention matrix, default rate is 0.1 from official paper
        padding_mask: mask for attention matrix for MLM
        attention_mask: mask for attention matrix for CLM
    Math:
        A = softmax(q•k^t/sqrt(D_h)), SA(z) = Av
    """
    attention_matrix = torch.matmul(q, k.transpose(-1, -2)) / dot_scale
    if padding_mask is not None:
        attention_matrix = attention_matrix.masked_fill(padding_mask == 0, float('-inf'))
    attention_dist = attention_dropout(
        F.softmax(attention_matrix, dim=-1)
    )
    attention_matrix = torch.matmul(attention_dist, v)
    return attention_matrix


class AttentionHead(nn.Module):
    """
    In this class, we implement workflow of single attention head in BERT
    This class has same role as Module "BertAttention" in official Repo (bert.py)
    Args:
        dim_model: dimension of model's latent vector space, default 1024 from official paper
        dim_head: dimension of each attention head, default 64 from official paper (1024 / 16)
        dropout: dropout rate for attention matrix, default 0.1 from official paper
    Math:
        A = softmax(attention Matrix/sqrt(3*D_h)), SA(z) = Av
    Reference:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/pdf/1810.04805.pdf
    """
    def __init__(self, dim_model: int = 1024, dim_head: int = 64, attention_dropout_prob: float = 0.1) -> None:
        super(AttentionHead, self).__init__()
        self.dim_model = dim_model
        self.dim_head = dim_head  # 1024 / 16 = 64
        self.dot_scale = torch.sqrt(torch.tensor(self.dim_head))
        self.attention_dropout = nn.Dropout(p=attention_dropout_prob)
        self.fc_q = nn.Linear(self.dim_model, self.dim_head)
        self.fc_k = nn.Linear(self.dim_model, self.dim_head)
        self.fc_v = nn.Linear(self.dim_model, self.dim_head)
        self.fc_qr = nn.Linear(self.dim_model, self.dim_head)  # projector for Relative Position Query matrix
        self.fc_kr = nn.Linear(self.dim_model, self.dim_head)  # projector for Relative Position Key matrix

    def forward(self, x: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tensor:
        q, k, v = self.fc_q(x), self.fc_k(x), self.fc_v(x)
        attention_matrix = scaled_dot_product_attention(
            q,
            k,
            v,
            self.dot_scale,
            self.attention_dropout,
            padding_mask,
            attention_mask
        )
        return attention_matrix


class MultiHeadAttention(nn.Module):
    """ In this class, we implement workflow of Multi-Head Self-attention for BERT
    This class has same role as Module "BertAttention" in official Repo (bert.py)
    In official repo, they use post-layer norm, but we use pre-layer norm which is more stable & efficient for training
    Args:
        dim_model: dimension of model's latent vector space, default 1024 from official paper
        num_attention_heads: number of heads in MHSA, default 16 from official paper for Transformer
        dim_head: dimension of each attention head, default 64 from official paper (1024 / 16)
        attention_dropout_prob: dropout rate, default 0.1
    Math:
        A = softmax(attention Matrix/sqrt(3*D_h)), SA(z) = Av
    Reference:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/pdf/1810.04805.pdf
    """
    def __init__(self, dim_model: int = 1024, num_attention_heads: int = 16, dim_head: int = 64, attention_dropout_prob: float = 0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        self.dim_model = dim_model
        self.num_attention_heads = num_attention_heads
        self.dim_head = dim_head
        self.attention_dropout_prob = attention_dropout_prob
        self.attention_heads = nn.ModuleList(
            [AttentionHead(self.dim_model, self.dim_head, self.attention_dropout_prob) for _ in range(self.num_attention_heads)]
        )
        self.fc_concat = nn.Linear(self.dim_model, self.dim_model)

    def forward(self, x: Tensor, padding_mask: Tensor, attention_mask: Tensor = None, emd: Tensor = None) -> Tensor:
        """ x is already passed nn.Layernorm """
        assert x.ndim == 3, f'Expected (batch, seq, hidden) got {x.shape}'
        attention_output = self.fc_concat(
            torch.cat([head(x, padding_mask, attention_mask, emd) for head in self.attention_heads], dim=-1)
        )
        return attention_output


class FeedForward(nn.Module):
    """
    Class for Feed-Forward Network module in Transformer Encoder Block, this module for BERT
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


class BERTEncoderLayer(nn.Module):
    """ Class for encoder model module in BERT
    In this class, we stack each encoder_model module (Multi-Head attention, Residual-Connection, LayerNorm, FFN)
    This class has same role as Module "BertEncoder" in official Repo (bert.py)
    In official repo, they use post-layer norm, but we use pre-layer norm which is more stable & efficient for training
    """
    def __init__(
            self,
            dim_model: int = 1024,
            num_attention_heads: int = 16,
            dim_ffn: int = 4096,
            layer_norm_eps: float = 0.02,
            attention_dropout_prob: float = 0.1,
            hidden_dropout_prob: float = 0.1
    ) -> None:
        super(BERTEncoderLayer, self).__init__()
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

    def forward(self, x: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tensor:
        """ rel_pos_emb is fixed for all layer in same forward pass time """
        ln_x = self.layer_norm1(x)
        residual_x = self.hidden_dropout(self.self_attention(ln_x, padding_mask, attention_mask)) + x
        ln_x = self.layer_norm2(residual_x)
        fx = self.ffn(ln_x) + residual_x
        return fx


class BERTEncoder(nn.Module, AbstractModel):
    """
    In this class, 1) encode input sequence, 2) make absolute position embedding,
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
            layer_norm_eps: float = 0.02,
            attention_dropout_prob: float = 0.1,
            hidden_dropout_prob: float = 0.1,
            gradient_checkpointing: bool = False
    ) -> None:
        super(BERTEncoder, self).__init__()
        self.cfg = cfg
        self.max_seq = max_seq
        self.num_layers = num_layers
        self.dim_model = dim_model
        self.num_attention_heads = num_attention_heads
        self.dim_ffn = dim_ffn
        self.hidden_dropout = nn.Dropout(p=hidden_dropout_prob)  # dropout is not learnable
        self.layer = nn.ModuleList(
            [BERTEncoderLayer(dim_model, num_attention_heads, dim_ffn, layer_norm_eps, attention_dropout_prob, hidden_dropout_prob) for _ in range(self.num_layers)]
        )
        self.layer_norm = nn.LayerNorm(dim_model, eps=layer_norm_eps)  # for final-Encoder output
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, inputs: Tensor, abs_pos_emb: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> tuple[Tensor, Tensor]:
        """ inputs: embedding from input sequence
        abs_pos_emb: absolute position embedding
        padding_mask: mask for Encoder padded token for speeding up to calculate attention score or MLM
        attention_mask: mask for CLM
        """
        layer_output = []
        x = inputs + abs_pos_emb  # add absolute position embedding with word embedding
        for layer in self.layer:
            if self.gradient_checkpointing and self.cfg.train:
                x = self._gradient_checkpointing_func(
                    layer.__call__,  # same as __forward__ call, torch reference recommend to use __call__ instead of forward
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
    """ BERT Embedding Module class
    This module has option => whether or not to use ALBERT Style Factorized Embedding
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
        self.word_embedding = nn.Embedding(len(cfg.tokenizer), cfg.dim_model)  # Word Embedding which is not add Absolute Position
        self.abs_pos_emb = nn.Embedding(cfg.max_seq, cfg.dim_model)  # Absolute Position Embedding for EMD Layer
        self.layer_norm1 = nn.LayerNorm(cfg.dim_model, eps=cfg.layer_norm_eps)  # for word embedding
        self.layer_norm2 = nn.LayerNorm(cfg.dim_model, eps=cfg.layer_norm_eps)  # for word embedding
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
        abs_pos_emb = self.hidden_dropout(
            self.layer_norm2(self.abs_pos_emb(torch.arange(inputs.shape[1]).repeat(inputs.shape[0]).view(inputs.shape[0], -1).to(inputs)))
        )  # "I" in paper)
        return word_embeddings, abs_pos_emb


class BERT(nn.Module, AbstractModel):
    """ Main class for BERT, having all of sub-blocks & modules such as self-attention, feed-forward, BERTEncoder ..
    Init Scale of BERT Hyper-Parameters, Embedding Layer, Encoder Blocks

    Args:
        cfg: configuration.CFG

    References:
        https://arxiv.org/pdf/1810.04805.pdf
    """
    def __init__(self, cfg: CFG) -> None:
        super(BERT, self).__init__()
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.max_seq = cfg.max_seq
        self.num_layers = cfg.num_layers
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
        self.encoder = BERTEncoder(
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
