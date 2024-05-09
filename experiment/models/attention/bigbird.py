import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment.models.abstract_model import AbstractModel
from torch import Tensor
from typing import Tuple, List
from einops.layers.torch import Rearrange
from configuration import CFG


def generalised_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dot_scale: Tensor,
    attention_dropout: nn.Dropout,
    padding_mask: Tensor = None,
    attention_mask: Tensor = None
) -> Tensor:
    """ generalised_attention from bigbird (google research)

    Generalised Attention = Random Attention + Window Attention + Global Attention

    Args:
        q: query matrix, shape (batch_size, seq_len, dim_head)
        k: key matrix, shape (batch_size, seq_len, dim_head)
        v: value matrix, shape (batch_size, seq_len, dim_head)
        dot_scale: scale factor for Q•K^T result
        attention_dropout: dropout for attention matrix, default rate is 0.1 from official paper
        padding_mask: mask for attention matrix for MLM, you must check whether or not padding token is 1
        attention_mask: mask for attention matrix for CLM

    Math:
        ATTND(X)i =xi +XσQh(xi)Kh(XN(i))T·Vh(XN(i))

    References:

    """
    pass
