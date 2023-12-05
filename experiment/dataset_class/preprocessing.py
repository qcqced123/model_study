import re, gc
import pandas as pd
import numpy as np
import torch
import configuration as configuration
from sklearn.model_selection import StratifiedKFold, GroupKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from nltk.tokenize import word_tokenize
from autocorrect import Speller
from spellchecker import SpellChecker
from tqdm.auto import tqdm
from typing import List

speller = Speller(lang='en')
spellchecker = SpellChecker()


def group_kfold(df: pd.DataFrame, cfg: configuration.CFG) -> pd.DataFrame:
    """ GroupKFold """
    fold = GroupKFold(
        n_splits=cfg.n_folds,
    )
    df['fold'] = -1
    for num, (tx, vx) in enumerate(fold.split(X=df, groups=df['prompt_id'])):
        df.loc[vx, "fold"] = int(num)
    return df


def stratified_kfold(df: pd.DataFrame, cfg: configuration.CFG) -> pd.DataFrame:
    """ GroupKFold """
    fold = StratifiedKFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.seed
    )
    df['fold'] = -1
    for num, (tx, vx) in enumerate(fold.split(X=df, y=df['prompt_id'])):
        df.loc[vx, "fold"] = int(num)
    return df


def mls_kfold(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """ Multilabel Stratified KFold """
    tmp_df = df.copy()
    y = pd.get_dummies(data=tmp_df.iloc[:, 2:8], columns=tmp_df.columns[2:8])
    fold = MultilabelStratifiedKFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.seed
    )
    for num, (tx, vx) in enumerate(fold.split(X=df, y=y)):
        df.loc[vx, "fold"] = int(num)
    del tmp_df
    gc.collect()
    return df


def add_target_token(cfg: configuration.CFG, token: str) -> None:
    """
    Add special token to pretrained tokenizer
    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        token: str, special token to add
    """
    special_token = token
    special_tokens_dict = {'additional_special_tokens': [f'{special_token}']}
    cfg.tokenizer.add_special_tokens(special_tokens_dict)
    tar_token_id = cfg.tokenizer(f'{special_token}', add_special_tokens=False)['input_ids'][0]

    setattr(cfg.tokenizer, 'tar_token', f'{special_token}')
    setattr(cfg.tokenizer, 'tar_token_id', tar_token_id)
    cfg.tokenizer.save_pretrained(f'{cfg.checkpoint_dir}/tokenizer/')


def add_anchor_token(cfg: configuration.CFG, token: str) -> None:
    """
    Add special token to pretrained tokenizer
    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        token: str, special token to add
    """
    special_token = token
    special_tokens_dict = {'additional_special_tokens': [f'{special_token}']}
    cfg.tokenizer.add_special_tokens(special_tokens_dict)
    anchor_token_id = cfg.tokenizer(f'{special_token}', add_special_tokens=False)['input_ids'][0]

    setattr(cfg.tokenizer, 'anchor_token', f'{special_token}')
    setattr(cfg.tokenizer, 'anchor_token_id', anchor_token_id)
    cfg.tokenizer.save_pretrained(f'{cfg.checkpoint_dir}/tokenizer/')


def tokenizing(cfg: configuration.CFG, text: str, padding: bool or str = 'max_length') -> any:
    """
    Preprocess text for LLM Input, for common batch system
    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        text: text from dataframe or any other dataset, please pass str type
        padding: padding options, default 'max_length'
                 if you want use smart batching, init this param to False
    """
    inputs = cfg.tokenizer.encode_plus(
        text,
        max_length=cfg.max_len,
        padding=padding,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False,  # later, we will add ourselves
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v)
    return inputs


def adjust_sequences(sequences: list, max_len: int):
    """
    Similar to dynamic padding concept
    Append slicing index from original, because original source code is implemented weired
    So it generates some problem for applying very longer sequence
    Add -1 value to slicing index, so we can get result what we want
    Args:
        sequences: list of each cell's token sequence in one unique notebook id, must pass tokenized sequence input_ids
        => sequences = [[1,2,3,4,5,6], [1,2,3,4,5,6], ... , [1,2,3,4,5]]
        max_len: max length of sequence into LLM Embedding Layer, default is 2048 for DeBERTa-V3-Large
    Reference:
         https://github.com/louis-she/ai4code/blob/master/ai4code/utils.py#L70
    """
    length_of_seqs = [len(seq) for seq in sequences]
    total_len = sum(length_of_seqs)
    cut_off = total_len - max_len
    if cut_off <= 0:
        return sequences, length_of_seqs

    for _ in range(cut_off):
        max_index = length_of_seqs.index(max(length_of_seqs))
        length_of_seqs[max_index] -= 1
    sequences = [sequences[i][:l-1] for i, l in enumerate(length_of_seqs)]
    return sequences, length_of_seqs


def subsequent_tokenizing(cfg: configuration.CFG, text: str) -> any:
    """
    Tokenize input sentence to longer sequence than common tokenizing
    Append padding strategy NOT Apply same max length, similar concept to dynamic padding
    Truncate longer sequence to match LLM max sequence
    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        text: text from dataframe or any other dataset, please pass str type
    Reference:
        https://www.kaggle.com/competitions/AI4Code/discussion/343714
        https://github.com/louis-she/ai4code/blob/master/tests/test_utils.py#L6
    """
    inputs = cfg.tokenizer.encode_plus(
        text,
        padding=False,
        truncation=False,
        return_tensors=None,
        add_special_tokens=False,  # No need to special token to subsequent text sequence
    )
    return inputs['input_ids']


def find_index(x: np.ndarray, value: np.ndarray) -> int:
    """
    Method for find some tensor element's index
    Args:
        x: tensor object, which is contained whole tensor elements
        value: element that you want to find index
    """
    tensor_index = int(np.where(x == value)[0])
    return tensor_index


def subsequent_decode(cfg: configuration.CFG, token_list: list) -> any:
    """
    Return decoded text from subsequent_tokenizing & adjust_sequences
    For making prompt text
    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        token_list: token list from subsequent_tokenizing & adjust_sequences
    """
    output = cfg.tokenizer.decode(token_list)
    return output


def sequence_length(cfg: configuration.CFG, text_list: list) -> list:
    """ Get sequence length of all text data for checking statistics value """
    length_list = []
    for text in tqdm(text_list):
        tmp_text = tokenizing(cfg, text)['attention_mask']
        length_list.append(tmp_text.count(1))
    return length_list


def spelling(text: str, spellchecker: SpellChecker) -> int:
    """ return number of mis-spelling words in original text """
    wordlist = text.split()
    amount_miss = len(list(spellchecker.unknown(wordlist)))
    return amount_miss


def add_spelling_dictionary(tokens: List[str], spellchecker: SpellChecker, speller: Speller) -> None:
    """dictionary update for py-spell checker and autocorrect"""
    spellchecker.word_frequency.load_words(tokens)
    speller.nlp_data.update({token: 1000 for token in tokens})


def spell_corrector(p_df: pd.DataFrame, s_df: pd.DataFrame, spellchecker: SpellChecker, speller: Speller) -> None:
    """ correct mis-spell words in summaries text """
    p_df['prompt_tokens'] = p_df['prompt_text'].apply(lambda x: word_tokenize(x))
    p_df['prompt_tokens'].apply(lambda x: add_spelling_dictionary(x, spellchecker, speller))
    s_df['fixed_text'] = s_df['text'].apply(lambda x: speller(x))


def check_null(df: pd.DataFrame) -> pd.Series:
    """ check if input dataframe has null type object...etc """
    return df.isnull().sum()


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data_folder from csv file like as train.csv, test.csv, val.csv
    """
    df = pd.read_csv(data_path)
    return df


def no_char(text):
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)
    text = re.sub(r"\^[a-zA-Z]\s+", " ", text)
    text = re.sub(r"\s+[a-zA-Z]$", " ", text)
    return text


def no_html_tags(text):
    return re.sub("<.*?>", " ", text)


def no_multi_spaces(text):
    return re.sub(r"\s+", " ", text, flags=re.I)


def underscore_to_space(text: str):
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    return text


def preprocess_text(source):
    # Remove all the special characters
    source = re.sub(r'\W', ' ', str(source))
    source = re.sub(r'^b\s+', '', source)
    source = source.lower()
    return source


def cleaning_words(text: str) -> str:
    """ Apply all of cleaning process to text data """
    tmp_text = no_html_tags(text)
    tmp_text = underscore_to_space(tmp_text)
    tmp_text = no_char(tmp_text)
    tmp_text = preprocess_text(tmp_text)
    tmp_text = no_multi_spaces(tmp_text)
    return tmp_text
