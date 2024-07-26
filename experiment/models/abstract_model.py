import functools
import torch.nn as nn

from configuration import CFG
from experiment.activation import activation
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Tuple, Union, Callable


class AbstractModel:
    """ Abstract Model Class for all models in this project
    Each model should inherit this class for using common functionalities
    Functions:
        1) set gradient checking point option
        2) set the post-attention layer design (ffn or glu) with hidden activation function
        3) set the normalization design (batch, layer, rms)
        4) set the dropout design (dropout, mixed out ...)
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

    def share_embedding(self) -> None:
        """ method for embedding share (E), there are 2 options:
            1) share embedding instances => share all of attributes of embedding instances
            2) share embedding weights => share only embedding weights
                - implementation's are quite different for each backbone model's architecture
        implementations are optional, so you can implement embedding share method above 2 options in your model
        """
        pass

    def select_post_attention_design(self, cfg: CFG):
        """method for design selecting between ffn(feed-forward network) or glu variants with hidden activation function

        this method will be called when user select the variants of glu
        """
        post_attn = None
        if cfg.hidden_act == "gelu":
            post_attn = "GEGLU"

        elif cfg.hidden_act == "relu":
            post_attn = "ReGLU"

        elif cfg.hidden_act == "swish":
            post_attn = "SwiGLUE"

        return getattr(activation, post_attn)(cfg.dim_model, cfg.dim_ffn)

    def select_normalization(self):
        """method for normalization design selecting between several options
        (batch-norm, layer-norm, rms-norm ...)
        """
        pass

    def select_dropout(self):
        """method for dropout design selecting between several options
        (batch-norm, layer-norm, rms-norm ...)
        """
        pass
