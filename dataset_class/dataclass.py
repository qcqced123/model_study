import pandas as pd
import torch
import configuration
from torch.utils.data import Dataset
from torch import Tensor
from typing import Dict, List, Tuple
from .name_rule import sentiment_analysis
from .preprocessing import tokenizing, unify_feature_name, cleaning_words
from .preprocessing import subsequent_tokenizing, adjust_sequences


class PretrainDataset(Dataset):
    """ Custom Dataset for Pretraining Task in NLP, such as MLM, CLM, ... etc

    if you select clm, dataset will be very long sequence of text, so this module will deal the text from the sliding window of the text
    you can use this module for generative model's pretrain task

    Also you must pass the input, which is already tokenized by tokenizer with cutting by model's max_length
    We recommend to use the full max_length inputs for the better performance like roberta, gpt2, gpt3 ...

    Args:
        inputs: inputs from tokenizing by tokenizer, which is a dictionary of input_ids, attention_mask, token_type_ids
        is_valid: if you want to use this dataset for validation, you can set this as True, default is False
    """
    def __init__(self, inputs: Dict, is_valid: bool = False) -> None:
        super().__init__()
        self.inputs = inputs
        self.input_ids = inputs['input_ids']
        self.is_valid = is_valid

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        batch_inputs = {}
        for k, v in self.inputs.items():
            # reduce memory usage by defending copying tensor
            batch_inputs[k] = torch.as_tensor(v[item]) if not self.is_valid else torch.as_tensor(v[item][0:2048])
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
        self.domain = self.df.get('domain', None).tolist()
        self.ratings = self.df.get('rating', None).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, item: int) -> Dict:
        cls_token = self.cfg.tokenizer.cls_token if self.cfg.tokenizer.cls_token is not None else '<s>'
        sep_token = self.cfg.tokenizer.sep_token if self.cfg.tokenizer.sep_token is not None else '</s>'

        prompt = ''
        prompt += cls_token + self.domain[item] + sep_token
        prompt += cleaning_words(self.title[item]) + sep_token
        prompt += cleaning_words(self.text[item]) + sep_token

        inputs = tokenizing(self.cfg, prompt, False, False)
        inputs['labels'] = torch.as_tensor(self.ratings[item] - 1)  # 1 ~ 5 -> 0 ~ 4
        return inputs


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


class TextSimilarityDataset(Dataset):
    """ Pytorch Dataset Module for Text Similarity Task in fine-tuning
    """
    def __init__(self) -> None:
        super().__init__()
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        pass


class TextSummarizationDataset(Dataset):
    """ Pytorch Dataset Module for Text Summarization Task in fine-tuning, such as CNN-DailyMail, XSum, ... etc
    """
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
