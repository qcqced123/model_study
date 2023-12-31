import functools
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Tuple, Union, Callable


class AbstractTask:
    """ Abstract model class for all tasks in this project
    Each task should inherit this class for using common functionalities
    Functions:
        1) Init Gradient Checkpointing Flag
        2) Weight Initialization
            - Pytorch Default Weight Initialization: He Initialization (Kaiming Initialization)
    """
    def __init__(self):
        super(AbstractTask, self).__init__()

    def _init_weights(self, module: nn.Module) -> None:
        """ over-ride initializes weights of the given module function for torch models
        you must implement this function in your task class
        Args:
            module (:obj:`torch.nn.Module`):
                The module to initialize weights for
        """
        """ over-ride initializes weights of the given module function (+initializes LayerNorm) """
        if isinstance(module, nn.Linear):
            if self.cfg.init_weight == 'normal':
                module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
            elif self.cfg.init_weight == 'xavier_uniform':
                module.weight.data = nn.init.xavier_uniform_(module.weight.data)
            elif self.cfg.init_weight == 'xavier_normal':
                module.weight.data = nn.init.xavier_normal_(module.weight.data)
            elif self.cfg.init_weight == 'kaiming_uniform':
                module.weight.data = nn.init.kaiming_uniform_(module.weight.data)
            elif self.cfg.init_weight == 'kaiming_normal':
                module.weight.data = nn.init.kaiming_normal_(module.weight.data)
            elif self.cfg.init_weight == 'orthogonal':
                module.weight.data = nn.init.orthogonal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            """ reference from torch.nn.Layernorm with elementwise_affine=True """
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
