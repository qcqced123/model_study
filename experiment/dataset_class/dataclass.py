import gc, ast, sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

from experiment.configuration import CFG
from experiment.dataset_class.preprocessing import tokenizing


class MLMDataset(Dataset):
    """ Custom Dataset for Masked Language Modeling
    Args:

    """
    def __init__(self) -> None:
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int) -> Dict:
        pass
