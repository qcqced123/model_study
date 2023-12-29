import functools
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Tuple, Union, Callable


class AbstractModel:
    """ Abstract Model Class for all models in this project
    Each model should inherit this class for using common functionalities
    Functions:
        1) Gradient Checkpointing
    """
    def __init__(self):
        super(AbstractModel, self).__init__()

    def _set_gradient_checkpointing(self, enable: bool = False, gradient_checkpointing_func: Callable = checkpoint):
        """ Set Flag for gradient checkpointing for the current model, and then apply it
        Args:
            model (:obj:`torch.nn.Module`):
                The model to enable gradient checkpointing for
            enable (:obj: bool, optional, default False):
                Whether or not to enable gradient checkpointing for the model.
            gradient_checkpointing_func:
                The gradient checkpointing function to use. Defaults to :obj:`torch.utils.checkpoint.checkpoint`.
        """
        is_gradient_checkpointing_set = False
        if hasattr(self, "gradient_checkpointing"):
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
                " `gradient_checkpointing` to modules of the model that uses checkpointing."
            )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the current model.
        Args:
            gradient_checkpointing_kwargs (dict, *optional*):
                Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
        """
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}

        gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)

    def gradient_checkpointing_disable(self):
        """ Deactivates gradient checkpointing for the current model
        """
        self._set_gradient_checkpointing(enable=False)
