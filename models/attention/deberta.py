import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List
from einops.layers.torch import Rearrange
from experiment.configuration import CFG


def build_relative_position(x_size: int) -> Tensor:
    """
    Build Relative Position Matrix for Disentangled Self-attention in DeBERTa
    Args:
        x_size: sequence length of query matrix
    Reference:
        https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/deberta/da_utils.py#L29
        https://arxiv.org/abs/2006.03654
    """
    x_index, y_index = torch.arange(x_size), torch.arange(x_size)  # same as rel_pos in official repo
    rel_pos = x_index.view(-1, 1) - y_index.view(1, -1)
    return rel_pos


def disentangled_attention(q: Tensor, k: Tensor, v: Tensor, qr: Tensor, kr: Tensor, attention_dropout: torch.nn.Dropout, mask: Tensor = None) -> Tensor:
    """
    Disentangled Self-attention for DeBERTa, same role as Module "DisentangledSelfAttention" in official Repo
    Args:
        q: content query matrix, shape (batch_size, seq_len, dim_head)
        k: content key matrix, shape (batch_size, seq_len, dim_head)
        v: content value matrix, shape (batch_size, seq_len, dim_head)
        qr: position query matrix, shape (batch_size, 2*max_relative_position, dim_head), r means relative position
        kr: position key matrix, shape (batch_size, 2*max_relative_position, dim_head), r means relative position
        attention_dropout: dropout for attention matrix, default rate is 0.1 from official paper
        mask: mask for attention matrix, shape (batch_size, seq_len, seq_len), apply before softmax layer
    Math:
        c2c = torch.matmul(q, k.transpose(-1, -2))  # A_c2c
        c2p = torch.gather(torch.matmul(q, kr.transpose(-1 z, -2)), dim=-1, index=c2p_pos)
        p2c = torch.gather(torch.matmul(qr, k.transpose(-1, -2)), dim=-2, index=c2p_pos)
        attention Matrix = c2c + c2p + p2c
        A = softmax(attention Matrix/sqrt(3*D_h)), SA(z) = Av
    Notes:
        dot_scale(range 1 ~ 3): scale factor for Q•K^T result, sqrt(3*dim_head) from official paper by microsoft,
        3 means that use full attention matrix(c2c, c2p, p2c), same as number of using what kind of matrix
        default 1, c2c is always used and c2p & p2c is optional
    References:
        https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/deberta/disentangled_attention.py
        https://arxiv.org/pdf/1803.02155.pdf
        https://arxiv.org/abs/2006.03654
        https://arxiv.org/abs/2111.09543
        https://arxiv.org/abs/1901.02860
        https://arxiv.org/abs/1906.08237
    """
    scale_factor = 1
    c2c = torch.matmul(q, k.transpose(-1, -2))  # A_c2c

    c2p_att = torch.matmul(q, kr.transpose(-1, -2))
    c2p_pos = build_relative_position(q.shape[1]) + kr.shape[1] / 2  # same as rel_pos in official repo
    c2p_pos = torch.clamp(c2p_pos, 0, kr.shape[1] - 1).repeat(q.shape[0], 1, 1)
    c2p = torch.gather(c2p_att, dim=-1, index=c2p_pos)
    if c2p is not None:
        scale_factor += 1

    p2c_att = torch.matmul(qr, k.transpose(-1, -2))
    p2c = torch.gather(p2c_att, dim=-2, index=c2p_pos)  # same as torch.gather(k•qr^t, dim=-1, index=c2p_pos)
    if p2c is not None:
        scale_factor += 1

    dot_scale = torch.sqrt(torch.tensor(scale_factor * q.shape[2]))  # from official paper by microsoft
    attention_matrix = (c2c + c2p + p2c) / dot_scale  # attention Matrix = A_c2c + A_c2r + A_r2c
    if mask is not None:
        attention_matrix = attention_matrix.masked_fill(mask == 0, float('-inf'))  # Padding Token Masking
    attention_dist = attention_dropout(
        F.softmax(attention_matrix, dim=-1)
    )
    attention_matrix = torch.matmul(attention_dist, v)
    return attention_matrix


class AttentionHead(nn.Module):
    """
    In this class, we implement workflow of single attention head in DeBERTa-Large
    This class has same role as Module "BertAttention" in official Repo (bert.py)
    Args:
        dim_model: dimension of model's latent vector space, default 1024 from official paper
        dim_head: dimension of each attention head, default 64 from official paper (1024 / 16)
        dropout: dropout rate for attention matrix, default 0.1 from official paper
    Math:
        attention Matrix = c2c + c2p + p2c
        A = softmax(attention Matrix/sqrt(3*D_h)), SA(z) = Av
    Reference:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/abs/2006.03654
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

    def forward(self, x: Tensor, pos_x: Tensor, mask: Tensor, emd: Tensor = None) -> Tensor:
        q, k, v, qr, kr = self.fc_q(x), self.fc_k(x), self.fc_v(x), self.fc_qr(pos_x), self.fc_kr(pos_x)
        if emd is not None:
            q = self.fc_q(emd)
        attention_matrix = disentangled_attention(q, k, v, qr, kr, self.attention_dropout, mask)
        return attention_matrix


class MultiHeadAttention(nn.Module):
    """
    In this class, we implement workflow of Multi-Head Self-attention for DeBERTa-Large
    This class has same role as Module "BertAttention" in official Repo (bert.py)
    In official repo, they use post-layer norm, but we use pre-layer norm which is more stable & efficient for training
    Args:
        dim_model: dimension of model's latent vector space, default 1024 from official paper
        num_attention_heads: number of heads in MHSA, default 16 from official paper for Transformer
        dim_head: dimension of each attention head, default 64 from official paper (1024 / 16)
        attention_dropout_prob: dropout rate, default 0.1
    Math:
        attention Matrix = c2c + c2p + p2c
        A = softmax(attention Matrix/sqrt(3*D_h)), SA(z) = Av
    Reference:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/abs/2006.03654
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

    def forward(self, x: Tensor, rel_pos_emb: Tensor, mask: Tensor, emd: Tensor = None) -> Tensor:
        """ x is already passed nn.Layernorm """
        assert x.ndim == 3, f'Expected (batch, seq, hidden) got {x.shape}'
        attention_output = self.fc_concat(
            torch.cat([head(x, rel_pos_emb, mask, emd) for head in self.attention_heads], dim=-1)
        )
        return attention_output


class FeedForward(nn.Module):
    """
    Class for Feed-Forward Network module in Transformer Encoder Block, this module for DeBERTa-Large
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


class DeBERTaEncoderLayer(nn.Module):
    """
    Class for encoder model module in DeBERTa-Large
    In this class, we stack each encoder_model module (Multi-Head attention, Residual-Connection, LayerNorm, FFN)
    This class has same role as Module "BertEncoder" in official Repo (bert.py)
    In official repo, they use post-layer norm, but we use pre-layer norm which is more stable & efficient for training
    References:
        https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/deberta/bert.py
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
        super(DeBERTaEncoderLayer, self).__init__()
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

    def forward(self, x: Tensor, pos_x: torch.nn.Embedding, mask: Tensor, emd: Tensor = None) -> Tensor:
        """ rel_pos_emb is fixed for all layer in same forward pass time """
        ln_x, ln_pos_x = self.layer_norm1(x), self.layer_norm1(pos_x)  # pre-layer norm, weight share
        residual_x = self.hidden_dropout(self.self_attention(ln_x, ln_pos_x, mask, emd)) + x
        ln_x = self.layer_norm2(residual_x)
        fx = self.ffn(ln_x) + residual_x
        return fx


class DeBERTaEncoder(nn.Module):
    """
    In this class, 1) encode input sequence, 2) make relative position embedding, 3) stack num_layers DeBERTaEncoderLayer
    This class's forward output is not integrated with EMD Layer's output
    Output have ONLY result of disentangled self-attention
    All ops order is from official paper & repo by microsoft, but ops operating is slightly different,
    Because they use custom ops, e.g. XDropout, XSoftmax, ..., we just apply pure pytorch ops
    Args:
        max_seq: maximum sequence length, named "max_position_embedding" in official repo, default 512, in official paper, this value is called 'k'
        num_layers: number of EncoderLayer, default 24 for large model
    Notes:
        self.rel_pos_emb: P in paper, this matrix is fixed during forward pass in same time, all layer & all module must share this layer from official paper
    References:
        https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/deberta/ops.py
    """
    def __init__(
            self,
            max_seq: int = 512,
            num_layers: int = 24,
            dim_model: int = 1024,
            num_attention_heads: int = 16,
            dim_ffn: int = 4096,
            layer_norm_eps: float = 0.02,
            attention_dropout_prob: float = 0.1,
            hidden_dropout_prob: float = 0.1
    ) -> None:
        super(DeBERTaEncoder, self).__init__()
        self.max_seq = max_seq
        self.num_layers = num_layers
        self.dim_model = dim_model
        self.num_attention_heads = num_attention_heads
        self.dim_ffn = dim_ffn
        self.hidden_dropout = nn.Dropout(p=hidden_dropout_prob)  # dropout is not learnable
        self.encoder_layers = nn.ModuleList(
            [DeBERTaEncoderLayer(dim_model, num_attention_heads, dim_ffn, layer_norm_eps, attention_dropout_prob, hidden_dropout_prob) for _ in range(self.num_layers)]
        )
        self.layer_norm = nn.LayerNorm(dim_model, eps=layer_norm_eps)  # for final-Encoder output

    def forward(self, inputs: Tensor, rel_pos_emb: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        inputs: embedding from input sequence
        rel_pos_emb: relative position embedding
        mask: mask for Encoder padded token for speeding up to calculate attention score
        """
        layer_output = []
        x, pos_x = inputs, rel_pos_emb  # x is same as word_embeddings or embeddings in official repo
        for layer in self.encoder_layers:
            x = layer(x, pos_x, mask)
            layer_output.append(x)
        last_hidden_state = self.layer_norm(x)  # because of applying pre-layer norm
        hidden_states = torch.stack(layer_output, dim=0).to(x.device)  # shape: [num_layers, BS, SEQ_LEN, DIM_Model]
        return last_hidden_state, hidden_states


class EnhancedMaskDecoder(nn.Module):
    """
    Class for Enhanced Mask Decoder module in DeBERTa, which is used for Masked Language Model (Pretrain Task)
    Word 'Decoder' means that denoise masked token by predicting masked token
    In official paper & repo, they might use 2 EMD layers for MLM Task
        First-EMD layer: query input == Absolute Position Embedding
        Second-EMD layer: query input == previous EMD layer's output
    And this layer's key & value input is output from last disentangled self-attention encoder layer,
    Also, all of them can share parameters and this layer also do disentangled self-attention
    In official repo, they implement this layer so hard coding that we can't understand directly & easily
    So, we implement this layer with our own style, as closely as possible to paper statement
    Notes:
        Also we temporarily implement only extract token embedding, not calculating logit, loss for MLM Task yet
        MLM Task will be implemented ASAP
    Args:
        encoder: list of nn.ModuleList, which is (num_emd * last encoder layer) from DeBERTaEncoder
    References:
        https://arxiv.org/abs/2006.03654
        https://arxiv.org/abs/2111.09543
        https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/apps/models/masked_language_model.py
    """
    def __init__(self, encoder: List[nn.ModuleList], dim_model: int = 1024, layer_norm_eps: float = 1e-7) -> None:
        super(EnhancedMaskDecoder, self).__init__()
        self.emd_layers = encoder
        self.layer_norm = nn.LayerNorm(dim_model, eps=layer_norm_eps)

    def emd_context_layer(self, hidden_states, abs_pos_emb, rel_pos_emb, mask):
        outputs = []
        query_states = hidden_states + abs_pos_emb  # "I" in official paper,
        for emd_layer in self.emd_layers:
            query_states = emd_layer(x=hidden_states, pos_x=rel_pos_emb, mask=mask, emd=query_states)
            outputs.append(query_states)
        last_hidden_state = self.layer_norm(query_states)  # because of applying pre-layer norm
        hidden_states = torch.stack(outputs, dim=0).to(hidden_states.device)
        return last_hidden_state, hidden_states

    def forward(self, hidden_states: Tensor, abs_pos_emb: Tensor, rel_pos_emb, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        hidden_states: output from last disentangled self-attention encoder layer
        abs_pos_emb: absolute position embedding
        rel_pos_emb: relative position embedding
        """
        last_hidden_state, emd_hidden_states = self.emd_context_layer(hidden_states, abs_pos_emb, rel_pos_emb, mask)
        return last_hidden_state, emd_hidden_states


class Embedding(nn.Module):
    """ DeBERTa Embedding Module class
    This Module set & initialize 3 Embedding Layers:
        1) Word Embedding. 2) Relative Positional Embedding, 3) Absolute Positional Embedding
    Args:
        cfg: configuration.py
    Notes:
        Absolute Positional Embedding does not add at bottom layers
        It will be added in top layer of encoder before hidden states are passed into Enhanced Mask Decoder
        And Relative Positional Embedding also doesn't add with Word Embedding
    """
    def __init__(self, cfg: CFG) -> None:
        super(Embedding, self).__init__()
        self.max_seq = cfg.max_seq
        self.max_rel_pos = 2 * self.max_seq
        self.word_embedding = nn.Embedding(cfg.vocab_size, cfg.dim_model)  # Word Embedding which is not add Absolute Position
        self.rel_pos_emb = nn.Embedding(self.max_rel_pos, cfg.dim_model)  # Relative Position Embedding
        self.abs_pos_emb = nn.Embedding(cfg.max_seq, cfg.dim_model)  # Absolute Position Embedding for EMD Layer
        self.layer_norm1 = nn.LayerNorm(cfg.dim_model, eps=cfg.layer_norm_eps)  # for word embedding
        self.layer_norm2 = nn.LayerNorm(cfg.dim_model, eps=cfg.layer_norm_eps)  # for rel_pos_emb
        self.hidden_dropout = nn.Dropout(p=cfg.hidden_dropout_prob)

    def forward(self, inputs: Tensor) -> Tuple[nn.Embedding, nn.Embedding, nn.Embedding]:
        word_embeddings = self.hidden_dropout(
            self.layer_norm1(self.word_embedding(inputs))
        )
        rel_pos_emb = self.hidden_dropout(
            self.layer_norm2(self.rel_pos_emb(torch.arange(self.max_rel_pos).repeat(inputs.shape[0]).to(inputs)))
        )
        abs_pos_emb = self.abs_pos_emb(torch.arange(self.max_seq).repeat(inputs.shape[0]).to(inputs))  # "I" in paper
        return word_embeddings, rel_pos_emb, abs_pos_emb


class DeBERTa(nn.Module):
    """
    Main class for DeBERTa, having all of sub-blocks & modules such as Disentangled Self-attention, DeBERTaEncoder, EMD
    Init Scale of DeBERTa Hyper-Parameters, Embedding Layer, Encoder Blocks, EMD Blocks
    And then make 3-types of Embedding Layer, Word Embedding, Absolute Position Embedding, Relative Position Embedding
    Var:
        vocab_size: size of vocab in DeBERTa's Native Tokenizer
        max_seq: maximum sequence length
        max_rel_pos: max_seq x2 for build relative position embedding
        num_layers: number of Disentangled-Encoder layers
        num_attention_heads: number of attention heads
        num_emd: number of EMD layers
        dim_model: dimension of model
        num_attention_heads: number of heads in multi-head attention
        dim_ffn: dimension of feed-forward network, same as intermediate size in official repo
        hidden_dropout_prob: dropout rate for embedding, hidden layer
        attention_probs_dropout_prob: dropout rate for attention
    Notes:
        MLM Task is not implemented yet, will be implemented ASAP, but you can get token encode output (embedding)
    References:
        https://arxiv.org/abs/2006.03654
        https://arxiv.org/abs/2111.09543
        https://github.com/microsoft/DeBERTa/blob/master/experiments/language_model/deberta_xxlarge.json
        https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/deberta/config.py
        https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/deberta/deberta.py
        https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/deberta/bert.py
        https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/deberta/disentangled_attention.py
        https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/apps/models/masked_language_model.py
    """
    def __init__(self, cfg: CFG) -> None:
        super(DeBERTa, self).__init__()
        # Init Scale of DeBERTa Module
        self.vocab_size = cfg.vocab_size
        self.max_seq = cfg.max_seq
        self.max_rel_pos = 2 * self.max_seq
        self.num_layers = cfg.num_layers
        self.num_attention_heads = cfg.num_attention_heads
        self.num_emd = cfg.num_emd
        self.dim_model = cfg.dim_model
        self.dim_ffn = cfg.dim_ffn
        self.layer_norm_eps = cfg.layer_norm_eps
        self.hidden_dropout_prob = cfg.hidden_dropout_prob
        self.attention_dropout_prob = cfg.attention_probs_dropout_prob

        # Init Embedding Layer
        self.embedding = Embedding(cfg)

        # Init Encoder Blocks & Modules
        self.encoder = DeBERTaEncoder(
            self.max_seq,
            self.num_layers,
            self.dim_model,
            self.num_attention_heads,
            self.dim_ffn,
            self.layer_norm_eps,
            self.attention_dropout_prob,
            self.hidden_dropout_prob,
        )
        self.emd_layers = [self.encoder.encoder_layers[-1] for _ in range(self.num_emd)]
        self.emd_encoder = EnhancedMaskDecoder(self.emd_layers, self.dim_model, self.layer_norm_eps)

    def forward(self, inputs: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        assert inputs.ndim == 3, f'Expected (batch, sequence, vocab_size) got {inputs.shape}'
        # Embedding Layer
        word_embeddings, rel_pos_emb, abs_pos_emb = self.embedding(inputs)

        # Disentangled Self-attention Encoder Layer
        last_hidden_state, hidden_states = self.encoder(word_embeddings, rel_pos_emb, mask)

        # Enhanced Mask Decoder Layer
        emd_hidden_states = hidden_states[-2]
        emd_last_hidden_state, emd_hidden_states = self.emd_encoder(emd_hidden_states, abs_pos_emb, rel_pos_emb, mask)
        return last_hidden_state, hidden_states, emd_last_hidden_state, emd_hidden_states