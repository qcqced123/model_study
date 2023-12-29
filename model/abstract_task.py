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
        pass
