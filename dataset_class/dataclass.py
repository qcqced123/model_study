import pandas as pd
import torch
import configuration
from torch.utils.data import Dataset
from torch import Tensor
from typing import Dict, List, Tuple
from name_rule import sentiment_analysis
from preprocessing import tokenizing, unify_feature_name, cleaning_words
from preprocessing import subsequent_tokenizing, adjust_sequences


class PretrainDataset(Dataset):
    """ Custom Dataset for Pretraining Task in NLP, such as MLM, CLM, ... etc

    Args:
        inputs: inputs from tokenizing by tokenizer, which is a dictionary of input_ids, attention_mask, token_type_ids
    """
    def __init__(self, inputs: Dict) -> None:
        super().__init__()
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
    """ Pytorch Dataset Module for Sentiment Analysis Task in fine-tuning

    Baseline Dataset for Sentiment Analysis Task is Amazon Review Dataset,

    ASAP, we will extend this module for more general sentiment analysis dataset
    such as Yelp, IMDB, and so on

    Args:
        cfg: configuration file for the experiment
        df: pd.DataFrame for the dataset

    Returns:
        inputs: dictionary of input_ids, attention_mask, token_type_ids from huggingface tokenizer
        labels: tensor of labels
    """
    def __init__(self, cfg: configuration.CFG, df: pd.DataFrame) -> None:
        super().__init__()
        self.cfg = cfg
        self.name_rule = sentiment_analysis.name_dict
        self.df = unify_feature_name(df, self.name_rule)
        self.text = self.df.get('text').tolist()
        self.title = self.df.get('title', None).tolist()
        self.label = self.df.get('rating').tolist()

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, item: int) -> Tuple[Dict, Tensor]:
        cls_token, sep_token = self.cfg.tokenizer.cls_token, self.cfg.tokenizer.sep_token

        prompt = ''
        prompt += cls_token + self.title[item].apply(cleaning_words) + sep_token
        prompt += self.text[item].apply(cleaning_words) + sep_token

        inputs = tokenizing(self.cfg, prompt, False)
        labels = torch.as_tensor(self.label[item])
        return inputs, labels


class QuestionAnsweringDataset(Dataset):
    """ Pytorch Dataset Module for QuestionAnswering Task in fine-tuning
    """

    def __init__(self) -> None:
        super().__init__()
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        pass


class TextGenerationDataset(Dataset):
    """ Pytorch Dataset Module for Text Generation Task in fine-tuning
    """
    def __init__(self) -> None:
        super().__init__()
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        pass


class TextSimilarityDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        pass


class TextSummationDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        pass


class SuperGlueDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        pass


class SquadDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        pass
