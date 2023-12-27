import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Tuple, Union


class CLMCollator(nn.Module):
    pass


class CLMHead(nn.Module):
    """
    Custom Casual Language Model Head for CLM Task, which is used for pre-training Auto-Regressive Model (AR)
    For Decoder, Such as GPT2, GPTNeo, ...
    Args:
        cfg: configuration.CFG
    References:
        https://huggingface.co/docs/transformers/main/tasks/language_modeling.html
    """