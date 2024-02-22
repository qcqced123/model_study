import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment.models.abstract_model import AbstractModel
from torch import Tensor
from typing import Tuple, List
from einops.layers.torch import Rearrange
from configuration import CFG


class RoFormer(nn.Module, AbstractModel):
    """
    Main class for RoFormer, having all of sub-blocks & modules such as Disentangled Self-attention, Encoder
    Init Scale of RoFormer Hyper-Parameters, Embedding Layer, Encoder Blocks
    And then make 3-types of Embedding Layer, Word Embedding, Absolute Position Embedding, Relative Position Embedding

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
        hidden_dropout_prob: dropout rate for embedding, hidden layer
        attention_probs_dropout_prob: dropout rate for attention

    Concepts:
        Multiple rotary embedding to context vector (Query, Key)

    Maths:
        fq(xm,m) = (Wq*xm)e**imθ
        fk(xn, n) = (Wk*xn)e**inθ

    References:

    """
    pass

