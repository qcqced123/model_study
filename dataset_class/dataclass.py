import torch
import configuration
from torch.utils.data import Dataset
from torch import Tensor
from typing import Dict, List, Tuple


class PretrainDataset(Dataset):
    """ Custom Dataset for Pre-

    Args:
        inputs: inputs from tokenizing by tokenizer, which is a dictionary of input_ids, attention_mask, token_type_ids
    """
    def __init__(self, inputs: Dict) -> None:
        self.inputs = inputs
        self.input_ids = inputs['input_ids']

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        batch_inputs = {}
        for k, v in self.inputs.items():
            batch_inputs[k] = torch.as_tensor(v[item])  # reduce memory usage by defending copying tensor
        return batch_inputs


class SentimentAnalysisDataset(Dataset):
    pass


class QuestionAnsweringDataset(Dataset):
    pass


class TextGenerationDataset(Dataset):
    pass


class TextSimilarityDataset(Dataset):
    pass


class TextSummationDataset(Dataset):
    pass


class SuperGlueDataset(Dataset):
    pass


class SQuAdDataset(Dataset):
    pass
