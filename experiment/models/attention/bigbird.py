import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment.models.abstract_model import AbstractModel
from torch import Tensor
from typing import Tuple, List
from einops.layers.torch import Rearrange
from configuration import CFG


def global_attention():
    """ function for full attention, applying selected index of token (Task-specific special tokens)
    such as [CLS], [SEP] ...

    this function have same idea with longformer global attention

    Args:

    Returns:

    Reference:

    """
    pass


def sliding_window_attention():
    """ function for sliding window (convolution) attention

    this function have same idea with longformer sliding window attention too

    Args:

    Returns:

    Reference:

    """
    pass


def random_attention():
    """ function for random attention, which is used in bigbird from google research

    Args:

    Returns:

    Reference:

    """
    pass

