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


def _set_gradient_checkpointing(model: nn.Module, enable: bool = True, gradient_checkpointing_func: Callable = checkpoint):
    """ Set Flag for gradient checkpointing for the current model, and then apply it
    Args:
        model (:obj:`torch.nn.Module`):
            The model to enable gradient checkpointing for
        enable (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to enable gradient checkpointing for the model.
        gradient_checkpointing_func:
            The gradient checkpointing function to use. Defaults to :obj:`torch.utils.checkpoint.checkpoint`.
    """
    is_gradient_checkpointing_set = False
    if hasattr(model, "gradient_checkpointing"):
        model._gradient_checkpointing_func = gradient_checkpointing_func
        model.gradient_checkpointing = enable
        is_gradient_checkpointing_set = True

    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module._gradient_checkpointing_func = gradient_checkpointing_func
            module.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True

    if not is_gradient_checkpointing_set:
        raise ValueError(
            f"{model.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
            " `gradient_checkpointing` to modules of the model that uses checkpointing."
        )


def gradient_checkpointing_enable(model: nn.Module, gradient_checkpointing_kwargs=None):
    """
    Activates gradient checkpointing for the current model.
    Args:
        model (:obj:`torch.nn.Module`):
            The model to enable gradient checkpointing for
        gradient_checkpointing_kwargs (dict, *optional*):
            Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
    """
    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}

    gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)
    _set_gradient_checkpointing(model, enable=True, gradient_checkpointing_func=gradient_checkpointing_func)


def gradient_checkpointing_disable(model: nn.Module):
    """ Deactivates gradient checkpointing for the current model
    Args:
        model (:obj:`torch.nn.Module`):
            The model to enable gradient checkpointing for
    """
    _set_gradient_checkpointing(model, enable=False)


@property
def is_gradient_checkpointing(model: nn.Module) -> bool:
    """ Whether gradient checkpointing is activated for this model or not.
    Args:
        model (:obj:`torch.nn.Module`):
            The model to enable gradient checkpointing for
    """
    return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in model.modules())

