import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment.models.abstract_model import AbstractModel
from torch import Tensor
from typing import Tuple, List
from einops.layers.torch import Rearrange
from configuration import CFG


class GPT2(nn.Module, AbstractModel):
    """ Main class for gpt2, which is same model architecture from vanilla transformers decoder, gpt1
    but this version of gpt model use pre-layer normalization method
    x1, x2, x3, x4 ... xt
    x1, x2, x3, x4, [mask] ... [mask]
    Args:

    Notes:

    References:

    """