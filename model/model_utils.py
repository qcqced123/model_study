import logging
import inspect
import functools
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch import Tensor
from typing import Callable, List


logger = logging.getLogger(__name__)


def freeze(module: nn.Module) -> None:
    """ Freezes module's parameters.
    Examples:
        freezing embeddings and first 2 layers of encoder
        1) freeze(model.embeddings
        2) freeze(model.encoder.layer[:2])
    """
    for parameter in module.parameters():
        parameter.requires_grad = False


def get_freeze_parameters(module: nn.Module) -> List[str]:
    """ Returns names of freezed parameters of the given module.
    Examples:
        freezed_parameters = get_freezed_parameters(model)
    """
    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)
    return freezed_parameters


def init_weights(auto_cfg, module: nn.Module) -> None:
    """ Initializes weights of the given module.
    """
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=auto_cfg.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=auto_cfg.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def reinit_topk(model: nn.Module, num_layers: int) -> None:
    """ Re-initialize the last-k transformer Encoder layers.
    Encoder Layer: Embedding, Attention Head, LayerNorm, Feed Forward
    Args:
        model: The target transformer model.
        num_layers: The number of layers to be re-initialized.
    """
    if num_layers > 0:
        model.encoder.layer[-num_layers:].apply(model._init_weights)