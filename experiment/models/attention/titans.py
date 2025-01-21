import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple, List
from einops.layers.torch import Rearrange
from transformers import AutoConfig, AutoTokenizer
from experiment.models.abstract_model import AbstractModel


def limited_window_attention():
    """ local window(sliding window, block-sparse, ...) attention func for encoding the "short-term memory" for Titans
    """
    return


class MultiHeadAttention(nn.Module):
    """
    """
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.attention = limited_window_attention

    def forward(self):
        return


class ShortTermMemory(nn.Module):
    """ module that is responsible to store/remember/encode to "short-term memory" by using "limited window attention"
    """
    def __init__(self, num_heads: int = 16):
        super(ShortTermMemory, self).__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList(
            [MultiHeadAttention() for _ in range(self.num_heads)]
        )

    def forward(self):
        return


class LongTermMemory(nn.Module):
    """ module that is responsible to store/remember long past
    """
    def __init__(self):
        super(LongTermMemory, self).__init__()

    def forward(self):
        return


class PersistentMemory(nn.Module):
    """ module of learnable but date-independent parameters that encodes the knowledge about a task,
    role as meta/common-sense knowledge in human-being
    """
    def __init__(self):
        super(PersistentMemory, self).__init__()

    def forward(self):
        return


class NeuralMemory(nn.Module):
    """ long-term neural memory module, role as meta(pre-context, prompt) memory at "in-context learning"
    this module learn how to encode memories into parameters that will be properly utilised when "inference time".

    this module measure the "surprise" of an input with the gradient of the neural network
    with respect to the input in associative memory loss.

    the more "surprise", the more memorable data/abstraction/memory.
    the word "surprise" means that

    """
    def __init__(self):
        super(NeuralMemory, self).__init__()

    def forward(self):
        return


class Titans(nn.Module):
    """ interface module of Titans architecture from Google Research, implemented by pytorch
    Args:

    Reference:
        https://arxiv.org/pdf/2501.00663
    """
    def __init__(self):
        super(Titans, self).__init__()

    def forward(self):
        return