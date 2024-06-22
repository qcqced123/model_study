import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from configuration import CFG
from typing import Tuple, List, Callable
from einops.layers.torch import Rearrange
from experiment.models.abstract_model import AbstractModel


def lsh_attention():
    """ function for locality-sensitive hashing attention (LSH), which is used in reformer from google research

    workflow:
        1) apply Locality-Sensitive Hashing (LSH) bucketing to sequence of query & key (same tensor, weighted share)
        2) sort the bucketed sequence by hash value
        3) chunk the sorted sequence
        4) do self-attention with same bucketed sequence
1
    Args:

    Returns:

    Reference:
        https://arxiv.org/abs/2001.04451
        https://tech.scatterlab.co.kr/reformer-review/
        https://research.google/blog/reformer-the-efficient-transformer/
        https://www.youtube.com/watch?v=i4H0kjxrias&ab_channel=YannicKilcher
    """
    pass


def reversible_residual_block(x: Tensor, attention_func: Callable, mlp_func: Callable) -> Tensor:
    """ function for reversible residual, which is used in reformer from google research
    original concept is came from the paper "Reversible Residual Networks" by Gao Huang et al. (2017)

    input should be the concatenated tensor of torch.concat([h, h], dim=-1), h is came from previous layer

    make the input tensor (x) in two part by splitting the hidden_state dimension,
    and pass the x1, x2 separately to the given algorithm

    Math:
        y1 = x1 + Attention(x2)
        y2 = x2 + FeedForward(y1)

    Args:
        x (torch.Tensor): input tensor, shape [batch, seq, num_heads, dim_head]
        attention_func (Callable): attention function, such as LSH, sliding window, global, ...
        mlp_func (Callable): feed-forward function from transformer block module's member

    Returns:
        y (torch.Tensor): output tensor, shape [batch, seq, num_heads, dim_head]

    Reference:
        https://arxiv.org/abs/2001.04451
        https://arxiv.org/pdf/1707.04585
        https://tech.scatterlab.co.kr/reformer-review/
        https://research.google/blog/reformer-the-efficient-transformer/
        https://www.youtube.com/watch?v=i4H0kjxrias&ab_channel=YannicKilcher

    """
    x1, x2 = x.chunk(2, dim=-1)  # split the hidden_state dimension
    y1 = x1 + attention_func(x2)
    y2 = x2 + mlp_func(y1)
    return torch.cat([y1, y2], dim=-1)
