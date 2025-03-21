import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from configuration import CFG
from typing import Tuple, List
from einops.layers.torch import Rearrange
from experiment.models.abstract_model import AbstractModel

def build_relative_position(x_size: int) -> Tensor:
    """ func for building the relative position matrix for "dis-entangled self-attention" in DeBERTa
    Args:
        x_size: sequence length of query matrix

    Reference:
        https://arxiv.org/abs/2006.03654
        https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/deberta/da_utils.py#L29
    """
    x_index, y_index = torch.arange(x_size, device="cuda"), torch.arange(x_size, device="cuda")  # same as rel_pos in official repo
    rel_pos = x_index.view(-1, 1) - y_index.view(1, -1)
    return rel_pos


def disentangled_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    qr: Tensor,
    kr: Tensor,
    attention_dropout: torch.nn.Dropout,
    padding_mask: Tensor = None,
    attention_mask: Tensor = None
) -> Tensor:
    """ func for implementing the "dis-entangled self-attention" for DeBERTa, same role as Module "DisentangledSelfAttention" in official Repo
    Args:
        q: content query matrix, shape (batch_size, seq_len, dim_head)
        k: content key matrix, shape (batch_size, seq_len, dim_head)
        v: content value matrix, shape (batch_size, seq_len, dim_head)
        qr: position query matrix, shape (batch_size, 2*max_relative_position, dim_head), r means relative position
        kr: position key matrix, shape (batch_size, 2*max_relative_position, dim_head), r means relative position
        attention_dropout: dropout for attention matrix, default rate is 0.1 from official paper
        padding_mask: mask for attention matrix for MLM
        attention_mask: mask for attention matrix for CLM

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
        https://arxiv.org/pdf/1803.02155
        https://arxiv.org/abs/2006.03654
        https://arxiv.org/abs/2111.09543
        https://arxiv.org/abs/1901.02860
        https://arxiv.org/abs/1906.08237
        https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/deberta/disentangled_attention.py
    """
    BS, NUM_HEADS, SEQ_LEN, DIM_HEADS = q.shape
    _, MAX_REL_POS, _, _ = kr.shape

    scale_factor = 1
    c2c = torch.matmul(q, k)  # A_c2c, q: (BS, NUM_HEADS, SEQ_LEN, DIM_HEADS), k: (BS, NUM_HEADS, DIM_HEADS, SEQ_LEN)
    c2p_att = torch.matmul(q, kr.permute(0, 2, 3, 1).contiguous())

    c2p_pos = build_relative_position(SEQ_LEN) + MAX_REL_POS / 2  # same as rel_pos in official repo
    c2p_pos = torch.clamp(c2p_pos, 0, MAX_REL_POS - 1).repeat(BS, NUM_HEADS, 1, 1).long()
    c2p = torch.gather(c2p_att, dim=-1, index=c2p_pos)
    if c2p is not None:
        scale_factor += 1

    p2c_att = torch.matmul(qr, k)  # qr: (BS, NUM_HEADS, SEQ_LEN, DIM_HEADS), k: (BS, NUM_HEADS, DIM_HEADS, SEQ_LEN)
    p2c = torch.gather(p2c_att, dim=-2, index=c2p_pos)  # same as torch.gather(k•qr^t, dim=-1, index=c2p_pos)
    if p2c is not None:
        scale_factor += 1

    dot_scale = torch.sqrt(torch.tensor(scale_factor * DIM_HEADS))  # from official paper by microsoft
    attention_matrix = (c2c + c2p + p2c) / dot_scale  # attention Matrix = A_c2c + A_c2r + A_r2c
    if padding_mask is not None:
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # for broadcasting: shape (BS, 1, 1, SEQ_LEN)
        attention_matrix = attention_matrix.masked_fill(padding_mask == 1, float('-inf'))  # Padding Token Masking

    attention_dist = attention_dropout(
        F.softmax(attention_matrix, dim=-1)
    )
    attention_matrix = torch.matmul(attention_dist, v).permute(0, 2, 1, 3).reshape(-1, SEQ_LEN, NUM_HEADS*DIM_HEADS).contiguous()
    return attention_matrix


class MultiHeadAttention(nn.Module):
    """ in this class, we implement workflow of Multi-Head Self-attention for DeBERTa-Large

    this class has same role as Module "BertAttention" in official Repo (bert.py)
    in official repo, they use post-layer norm, but we use pre-layer norm which is more stable & efficient for training

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

    def __init__(
        self,
        dim_model: int = 1024,
        num_attention_heads: int = 12,
        dim_head: int = 64,
        attention_dropout_prob: float = 0.1
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.dim_model = dim_model
        self.num_attention_heads = num_attention_heads
        self.dim_head = dim_head
        self.fc_q = nn.Linear(self.dim_model, self.dim_model)
        self.fc_k = nn.Linear(self.dim_model, self.dim_model)
        self.fc_v = nn.Linear(self.dim_model, self.dim_model)
        self.fc_qr = nn.Linear(self.dim_model, self.dim_model)  # projector for Relative Position Query matrix
        self.fc_kr = nn.Linear(self.dim_model, self.dim_model)  # projector for Relative Position Key matrix
        self.fc_concat = nn.Linear(self.dim_model, self.dim_model)
        self.attention = disentangled_attention
        self.attention_dropout = nn.Dropout(p=attention_dropout_prob)

    def forward(
        self,
        x: Tensor,
        rel_pos_emb: Tensor,
        padding_mask: Tensor,
        attention_mask: Tensor = None,
        emd: Tensor = None
    ) -> Tensor:
        """ x is already passed nn.Layernorm """
        assert x.ndim == 3, f'Expected (batch, seq, hidden) got {x.shape}'

        # size: bs, seq, nums head, dim head, linear projection
        q = self.fc_q(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        k = self.fc_k(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head).permute(0, 2, 3, 1).contiguous()
        v = self.fc_v(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        qr = self.fc_qr(rel_pos_emb).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        kr = self.fc_kr(rel_pos_emb).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head)
        if emd is not None:
            q = self.fc_q(emd).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        attention_matrix = self.attention(
            q,
            k,
            v,
            qr,
            kr,
            self.attention_dropout,
            padding_mask,
            attention_mask
        )
        attention_output = self.fc_concat(attention_matrix)
        return attention_output


class FeedForward(nn.Module):
    """ module of feed-forward network module in Transformer encoder block, this module for DeBERTa Series
    same role as module "BertIntermediate" in official Repo (bert.py)

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
    """ module of DeBERTa's encoder layer,
    in this class, we stack each encoder_model module (Multi-Head attention, Residual-Connection, LayerNorm, FFN)

    this class has same role as Module "BertEncoder" in official Repo (bert.py)
    in official repo, they use post-layer norm, but we use pre-layer norm which is more stable & efficient for training

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

    def forward(self, x: Tensor, pos_x: torch.nn.Embedding, padding_mask: Tensor, attention_mask: Tensor = None, emd: Tensor = None) -> Tensor:
        """ rel_pos_emb is fixed for all layer in same forward pass time """
        ln_x, ln_pos_x = self.layer_norm1(x), self.layer_norm1(pos_x)  # pre-layer norm, weight share
        residual_x = self.hidden_dropout(self.self_attention(ln_x, ln_pos_x, padding_mask, attention_mask, emd)) + x
        ln_x = self.layer_norm2(residual_x)
        fx = self.ffn(ln_x) + residual_x
        return fx


class DeBERTaEncoder(nn.Module, AbstractModel):
    """ interface of DeBERTa's encoder, in this interface module have workflow below:
    1) encode input sequence
    2) make relative position embedding
    3) stack num_layers DeBERTaEncoderLayer

    this class's forward output is not integrated with emd's output
    output have ONLY result of disentangled self-attention

    all ops order is from official paper & repo by microsoft, but ops operating is slightly different,
    because they use custom ops, e.g. XDropout, XSoftmax, ..., we just apply pure pytorch ops

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
        cfg: CFG,
        max_seq: int = 512,
        num_layers: int = 24,
        dim_model: int = 1024,
        num_attention_heads: int = 16,
        dim_ffn: int = 4096,
        layer_norm_eps: float = 0.02,
        attention_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        gradient_checkpointing: bool = False
    ) -> None:
        super(DeBERTaEncoder, self).__init__()
        self.cfg = cfg
        self.max_seq = max_seq
        self.num_layers = num_layers
        self.dim_model = dim_model
        self.num_attention_heads = num_attention_heads
        self.dim_ffn = dim_ffn
        self.hidden_dropout = nn.Dropout(p=hidden_dropout_prob)  # dropout is not learnable
        self.layer = nn.ModuleList(
            [DeBERTaEncoderLayer(dim_model, num_attention_heads, dim_ffn, layer_norm_eps, attention_dropout_prob, hidden_dropout_prob) for _ in range(self.num_layers)]
        )
        self.layer_norm = nn.LayerNorm(dim_model, eps=layer_norm_eps)  # for final-Encoder output
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, inputs: Tensor, rel_pos_emb: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: embedding from input sequence
            rel_pos_emb: relative position embedding
            padding_mask: mask for Encoder padded token for speeding up to calculate attention score or MLM
            attention_mask: mask for CLM
        """
        layer_output = []
        x, pos_x = inputs, rel_pos_emb  # x is same as word_embeddings or embeddings in official repo
        for layer in self.layer:
            if self.gradient_checkpointing and self.cfg.train:
                x = self._gradient_checkpointing_func(
                    layer.__call__,  # same as __forward__ call, torch reference recommend to use __call__ instead of forward
                    x,
                    pos_x,
                    padding_mask,
                    attention_mask
                )
            else:
                x = layer(
                    x,
                    pos_x,
                    padding_mask,
                    attention_mask
                )
            layer_output.append(x)
        last_hidden_state = self.layer_norm(x)  # because of applying pre-layer norm
        hidden_states = torch.stack(layer_output, dim=0).to(x.device)  # shape: [num_layers, BS, SEQ_LEN, DIM_Model]
        return last_hidden_state, hidden_states


class EnhancedMaskDecoder(nn.Module, AbstractModel):
    """ module of enhanced mask decoder in DeBERTa, word 'Decoder' means that denoise masked token by predicting masked token
    for masked language modeling (MLM)

    in official paper & repo, they might use 2 EMD layers for MLM Task
        1) First-EMD layer: query input == Absolute Position Embedding
        2) Second-EMD layer: query input == previous EMD layer's output

    and this layer's key & value input is output from last disentangled self-attention encoder layer,
    also, all of them can share parameters and this layer also do disentangled self-attention

    in official repo, they implement this layer so hard coding that we can't understand directly & easily
    so, we implement this layer with our own style, as closely as possible to paper statement

    Args:
        encoder: list of nn.ModuleList, which is (num_emd * last encoder layer) from DeBERTaEncoder
        gradient_checkpointing: whether to use gradient checkpointing or not, default False

    References:
        https://arxiv.org/abs/2006.03654
        https://arxiv.org/abs/2111.09543
        https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/apps/models/masked_language_model.py
    """
    def __init__(
            self,
            cfg: CFG,
            encoder: List[nn.ModuleList],
            dim_model: int = 1024,
            layer_norm_eps: float = 1e-7,
            gradient_checkpointing: bool = False
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.emd_layers = encoder
        self.layer_norm = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.gradient_checkpointing = gradient_checkpointing

    def emd_context_layer(self, hidden_states: Tensor, abs_pos_emb: nn.Embedding, rel_pos_emb: nn.Embedding, padding_mask: Tensor, attention_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        outputs = []
        query_states = hidden_states + abs_pos_emb  # "I" in official paper
        for emd_layer in self.emd_layers:
            if self.gradient_checkpointing and self.cfg.train:
                query_states = self._gradient_checkpointing_func(
                    emd_layer.__call__,  # same as __forward__ call, torch reference recommend to use __call__ instead of forward
                    hidden_states,
                    rel_pos_emb,
                    padding_mask,
                    query_states
                )
            else:
                query_states = emd_layer(
                    x=hidden_states,
                    pos_x=rel_pos_emb,
                    padding_mask=padding_mask,
                    emd=query_states
                )
            outputs.append(query_states)
        last_hidden_state = self.layer_norm(query_states)  # because of applying pre-layer norm
        hidden_states = torch.stack(outputs, dim=0).to(hidden_states.device)
        return last_hidden_state, hidden_states

    def forward(self, hidden_states: Tensor, abs_pos_emb: nn.Embedding, rel_pos_emb, padding_mask: Tensor, attention_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            hidden_states: output from last disentangled self-attention encoder layer
            abs_pos_emb: absolute position embedding
            rel_pos_emb: relative position embedding
            padding_mask: mask for Encoder padded token for speeding up to calculate attention score or MLM
            attention_mask: mask for CLM
        """
        last_hidden_state, emd_hidden_states = self.emd_context_layer(
            hidden_states,
            abs_pos_emb,
            rel_pos_emb,
            padding_mask,
            attention_mask
        )
        return last_hidden_state, emd_hidden_states


class Embedding(nn.Module):
    """ module of initializing DeBERTa's three embedding vector:
    1) word embedding vector (content, context embedding vector)
    2) relative position embedding vector (vector for projecting to query-position, key-position latent vector space)
    3) absolute position embedding vector (vector for integrating with query latent matrix in enhanced mask decoder context)

    Args:
        cfg: configuration.py

    Notes:
        Absolute Positional Embedding does not add at bottom layers
        It will be added in top layer of encoder before hidden states are passed into Enhanced Mask Decoder
        And Relative Positional Embedding also doesn't add with Word Embedding
    """
    def __init__(self, cfg: CFG) -> None:
        super(Embedding, self).__init__()
        self.cfg = cfg
        self.max_seq = cfg.max_seq
        self.max_rel_pos = 2 * self.max_seq
        self.word_embedding = nn.Embedding(len(cfg.tokenizer), cfg.dim_model)  # Word Embedding which is not add Absolute Position
        self.rel_pos_emb = nn.Embedding(self.max_rel_pos, cfg.dim_model)  # Relative Position Embedding
        self.abs_pos_emb = nn.Embedding(cfg.max_seq, cfg.dim_model)  # Absolute Position Embedding for EMD Layer
        self.layer_norm1 = nn.LayerNorm(cfg.dim_model, eps=cfg.layer_norm_eps)  # for word embedding
        self.layer_norm2 = nn.LayerNorm(cfg.dim_model, eps=cfg.layer_norm_eps)  # for rel_pos_emb
        self.layer_norm3 = nn.LayerNorm(cfg.dim_model, eps=cfg.layer_norm_eps)  # for abs_pos_emb
        self.hidden_dropout = nn.Dropout(p=cfg.hidden_dropout_prob)

        # ALBERT Style Factorized Embedding
        if self.cfg.is_mf_embedding:
            self.word_embedding = nn.Embedding(len(cfg.tokenizer), int(cfg.dim_model/6))
            self.projector = nn.Linear(int(cfg.dim_model/6), cfg.dim_model)  # project to original hidden dim

    def forward(self, inputs: Tensor) -> Tuple[nn.Embedding, nn.Embedding, nn.Embedding]:
        if self.cfg.is_mf_embedding:
            word_embeddings = self.hidden_dropout(
                self.layer_norm1(self.projector(self.word_embedding(inputs)))
            )
        else:
            word_embeddings = self.hidden_dropout(
                self.layer_norm1(self.word_embedding(inputs))
            )
        rel_pos_emb = self.hidden_dropout(
            self.layer_norm2(self.rel_pos_emb(torch.arange(inputs.shape[1], device="cuda").repeat(inputs.shape[0]).view(inputs.shape[0], -1)))
        )
        abs_pos_emb = self.hidden_dropout(
            self.layer_norm3(self.abs_pos_emb(torch.arange(inputs.shape[1], device="cuda").repeat(inputs.shape[0]).view(inputs.shape[0], -1)))
        )  # "I" in paper)
        return word_embeddings, rel_pos_emb, abs_pos_emb


class DeBERTa(nn.Module, AbstractModel):
    """ interface(main class) for DeBERTa, having all of sub-blocks & modules such as dis-entangled self-attention, encoder, enhanced mask decoder
    this model have two unique techniques:

        1) dis-entangled self-attention
            - separate the context vector embedding and relative position vector embedding
            - attention query context and key context, query context and key position, query position and key context
            - attention matrix = c2c + c2p + p2c

        2) enhanced mask decoder
            - integrate the absolute position vector embedding and relative position vector embedding
            - last two encoder layer will be weight shared for enhanced mask decoder
            - almost workflows are the same before encoder layer, but before the do self-attention,
              absolute position vector embedding will be added to activation, projected to query latent space only

    Args:
        cfg: configuration.CFG
        num_layers: number of EncoderLayer, default 12 for base model
        this value must be init by user for objective task
        if you select electra, you should set num_layers twice (generator, discriminator)

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
    def __init__(self, cfg: CFG, num_layers: int = 12) -> None:
        super(DeBERTa, self).__init__()
        # Init Scale of DeBERTa Module
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.max_seq = cfg.max_seq
        self.max_rel_pos = 2 * self.max_seq
        self.num_layers = num_layers
        self.num_attention_heads = cfg.num_attention_heads
        self.num_emd = cfg.num_emd
        self.dim_model = cfg.dim_model
        self.dim_ffn = cfg.dim_ffn
        self.layer_norm_eps = cfg.layer_norm_eps
        self.hidden_dropout_prob = cfg.hidden_dropout_prob
        self.attention_dropout_prob = cfg.attention_probs_dropout_prob
        self.gradient_checkpointing = cfg.gradient_checkpoint

        # Init Embedding Layer
        self.embeddings = Embedding(cfg)

        # Init Encoder Blocks & Modules
        self.encoder = DeBERTaEncoder(
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
        self.emd_layers = [self.encoder.layer[-1] for _ in range(self.num_emd)]
        self.emd_encoder = EnhancedMaskDecoder(
            self.cfg,
            self.emd_layers,
            self.dim_model,
            self.layer_norm_eps,
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

        word_embeddings, rel_pos_emb, abs_pos_emb = self.embeddings(inputs)
        last_hidden_state, hidden_states = self.encoder(
            word_embeddings,
            rel_pos_emb,
            padding_mask,
            attention_mask
        )

        emd_hidden_states = hidden_states[-self.cfg.num_emd]
        emd_last_hidden_state, emd_hidden_states = self.emd_encoder(
            emd_hidden_states,
            abs_pos_emb,
            rel_pos_emb,
            padding_mask,
            attention_mask
        )
        return emd_last_hidden_state, emd_hidden_states
