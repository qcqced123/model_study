import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from typing import Dict, List, Tuple, Union, Optional, Any
from configuration import CFG


class RTDCollator(nn.Module):
    """ Replaced Token Detection Collator (RTD) for Pretraining
    from ELECTRA original paper

    """