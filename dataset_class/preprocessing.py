import re, gc, pickle
import pandas as pd
import numpy as np
import torch
import configuration as configuration
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from autocorrect import Speller
from spellchecker import SpellChecker
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Callable, Any


speller = Speller(lang='en')
spellchecker = SpellChecker()


def hf_load_dataset(cfg: configuration.CFG) -> DatasetDict:
    """ Load dataset from Huggingface Datasets

    Notes:
        This function is temporary just fit-able for Wikipedia dataset

    References:
        https://github.com/huggingface/datasets/blob/main/src/datasets/load.py#2247
    """
    dataset = load_dataset(cfg.hf_dataset, cfg.language)
    return dataset


def hf_split_dataset(cfg: configuration.CFG, dataset: Dataset) -> Tuple[Dataset, Dataset]:
    """ Split dataset from Huggingface Datasets with huggingface method "train_test_split"

    Args:
        cfg: configuration.CFG, needed to load split ratio, seed value
        dataset: Huggingface Datasets object, dataset from Huggingface Datasets

    Notes:
        This function is temporary just fit-able for Wikipedia dataset & MLM Task
    """
    dataset = dataset.train_test_split(cfg.split_ratio, seed=cfg.seed)
    train, valid = dataset['train'], dataset['test']
    return train, valid


def dataset_split(cfg: configuration.CFG, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Split dataset from pandas.DataFrame with sklearn.train_test_split

    Args:
        cfg: configuration.CFG, needed to load split ratio, seed value
        df: pandas.DataFrame, dataset from csv file
    """
    train, valid = train_test_split(
        df,
        test_size=cfg.split_ratio,
        random_state=cfg.seed,
        shuffle=True,
    )
    return train, valid


def dict2df(dataset: Dict) -> pd.DataFrame:
    """ Convert dictionary to pandas.DataFrame
    """
    df = pd.DataFrame(dataset)
    return df


def group_kfold(df: pd.DataFrame, cfg: configuration.CFG) -> pd.DataFrame:
    """ Group K Fold by sklearn

    Args:
        df: pandas.DataFrame, dataset from csv file
        cfg: configuration.CFG, needed to load split ratio, seed value
    """
    fold = GroupKFold(
        n_splits=cfg.n_folds,
    )
    df['fold'] = -1
    for num, (tx, vx) in enumerate(fold.split(X=df, groups=df['prompt_id'])):
        df.loc[vx, "fold"] = int(num)
    return df


def stratified_kfold(df: pd.DataFrame, cfg: configuration.CFG) -> pd.DataFrame:
    """ Stratified K Fold by sklearn

    Args:
        df: pandas.DataFrame, dataset from csv file
        cfg: configuration.CFG, needed to load split ratio, seed value
    """
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
    """ Multilabel Stratified KFold by iterstrat

    Args:
        df: pandas.DataFrame, dataset from csv file
        cfg: configuration.CFG, needed to load split ratio, seed value
    """
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


def chunking(cfg: configuration.CFG, sequences: Dict) -> List[str]:
    """ Chunking sentence to token using pretrained tokenizer

    Args:
        cfg: configuration.CFG, needed to load pretrained tokenizer
        sequences: list, sentence to chunking

    References:
        https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling
    """
    return cfg.tokenizer([" ".join(x) for x in sequences['text']])


def group_texts(cfg: configuration.CFG, sequences: Dict) -> Dict:
    """ Dealing Problem: some of data instances are longer than the maximum input length for the model,
    This function is ONLY used to HF Dataset Object

    1) Concatenate all texts
    2) We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    3) customize this part to your needs
    4) Split by chunks of max_len

    """
    concatenated_sequences = {k: sum(sequences[k], []) for k in sequences.keys()}
    total_length = len(concatenated_sequences[list(sequences.keys())[0]])
    if total_length >= cfg.max_seq:
        total_length = (total_length // cfg.max_seq) * cfg.max_seq
    result = {
        k: [t[i: i + cfg.max_seq] for i in range(0, total_length, cfg.max_seq)]
        for k, t in concatenated_sequences.items()
    }
    return result


def apply_preprocess(dataset: Dataset, function: Callable, batched: bool = True, num_proc: int = 4, remove_columns: Any = None) -> Dataset:
    """ Apply preprocessing to text data, which is using huggingface dataset method "map()"
    for pretrained training (MLM, CLM)

    Args:
        dataset: Huggingface Datasets object, dataset from Huggingface Datasets
        function: Callable, function that you want to apply
        batched: bool, default True, if you want to apply function to batched data, set True
        num_proc: int, default 4, number of process for multiprocessing
        remove_columns: any, default None, if you want to remove some columns, set column name

    References:
        https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling
    """
    mapped_dataset = dataset.map(
        function,
        batched=batched,
        num_proc=num_proc,
        remove_columns=remove_columns,
    )
    return mapped_dataset


def tokenizing(cfg: configuration.CFG, text: str, padding: bool or str = 'max_length') -> Any:
    """ Preprocess text for LLM Input, for common batch system

    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        text: text from dataframe or any other dataset, please pass str type
        padding: padding options, default 'max_length', if you want use smart batching, init this param to False
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


def adjust_sequences(sequences: List, max_len: int):
    """ Similar to dynamic padding concept
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


def subsequent_tokenizing(cfg: configuration.CFG, text: str) -> Any:
    """ Tokenize input sentence to longer sequence than common tokenizing
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
    """ Method for find some tensor element's index

    Args:
        x: tensor object, which is contained whole tensor elements
        value: element that you want to find index
    """
    tensor_index = int(np.where(x == value)[0])
    return tensor_index


def subsequent_decode(cfg: configuration.CFG, token_list: List) -> Any:
    """ Return decoded text from subsequent_tokenizing & adjust_sequences
    For making prompt text

    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        token_list: token list from subsequent_tokenizing & adjust_sequences
    """
    output = cfg.tokenizer.decode(token_list)
    return output


def sequence_length(cfg: configuration.CFG, text_list: List) -> List:
    """ Get sequence length of all text data for checking statistics value
    """
    length_list = []
    for text in tqdm(text_list):
        tmp_text = tokenizing(cfg, text)['attention_mask']
        length_list.append(tmp_text.count(1))
    return length_list


def spelling(text: str, spellchecker: SpellChecker) -> int:
    """ return number of mis-spelling words in original

    Args:
        text: str, text data
        spellchecker: SpellChecker, spellchecker object from spellchecker library
    """
    wordlist = text.split()
    amount_miss = len(list(spellchecker.unknown(wordlist)))
    return amount_miss


def add_spelling_dictionary(tokens: List[str], spellchecker: SpellChecker, speller: Speller) -> None:
    """ dictionary update for py-spell checker and autocorrect

    Args:
        tokens: list of tokens, which is contained prompt text
        spellchecker: SpellChecker, spellchecker object from spellchecker library
        speller: Speller, speller object from autocorrect library
    """
    spellchecker.word_frequency.load_words(tokens)
    speller.nlp_data.update({token: 1000 for token in tokens})


def check_null(df: pd.DataFrame) -> pd.Series:
    """ check if input dataframe has null type object...etc
    """
    return df.isnull().sum()


def load_data(data_path: str) -> pd.DataFrame:
    """ Load data_folder from csv file like as train.csv, test.csv, val.csv
    """
    df = pd.read_csv(data_path)
    return df


def no_char(text):
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)
    text = re.sub(r"\^[a-zA-Z]\s+", " ", text)
    text = re.sub(r"\s+[a-zA-Z]$", " ", text)
    return text


def no_multi_spaces(text):
    return re.sub(r"\s+", " ", text, flags=re.I)


def underscore_to_space(text: str):
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    return text


def preprocess_text(source):
    """ Remove all the special characters
    """
    source = re.sub(r'\W', ' ', str(source))
    source = re.sub(r'^b\s+', '', source)
    source = source.lower()
    return source


def cleaning_words(text: str) -> str:
    """ Apply all of cleaning process to text data
    """
    tmp_text = underscore_to_space(text)
    tmp_text = no_char(tmp_text)
    tmp_text = preprocess_text(tmp_text)
    tmp_text = no_multi_spaces(tmp_text)
    return tmp_text


def split_token(inputs: str) -> List:
    """ Convert mal form list (ex. string list in pd.DataFrame) to Python List object & elementwise type casting
    """
    inputs = cleaning_words(inputs)
    tmp = inputs.split()
    result = list(map(int, tmp))
    return result


def split_list(inputs: List, max_length: int) -> List[List]:
    """ Split List into sub shorter list, which is longer than max_length
    """
    result = [inputs[i:i + max_length] for i in range(0, len(inputs), max_length)]
    return result


def flatten_sublist(inputs: List[List], max_length: int = 512) -> List[List]:
    """ Flatten Nested List to 1D-List """
    result = []
    for instance in tqdm(inputs):
        tmp = split_token(instance)
        if len(tmp) > max_length:
            tmp = split_list(tmp, max_length)
            for i in range(len(tmp)):
                result.append(tmp[i])
        else:
            result.append(tmp)
    return result


def preprocess4tokenizer(input_ids: List, token_type_ids: List, attention_mask: List):
    """ Preprocess function for handling exception in inputs data instance
    which is some of input_ids, token_type_ids, attention_mask are not started with [CLS] token or are not ended with [SEP] token
    """
    for i, inputs in tqdm(enumerate(input_ids)):
        if inputs[0] != 1:
            inputs.insert(0, 1)
            token_type_ids[i].insert(0, 0)
            attention_mask[i].insert(0, 1)
        if inputs[-1] != 2:
            inputs.append(2)
            token_type_ids[i].append(0)
            attention_mask[i].append(1)
    return input_ids, token_type_ids, attention_mask


def cut_instance(input_ids: List, token_type_ids: List, attention_mask: List, min_length: int = 256):
    """ Function for cutting instance which is shorter than min_length
    """
    n_input_ids, n_token_type_ids, n_attention_mask = [], [], []
    for i, inputs in tqdm(enumerate(input_ids)):
        if len(inputs) >= min_length:
            n_input_ids.append(inputs)
            n_token_type_ids.append(token_type_ids[i])
            n_attention_mask.append(attention_mask[i])
    return n_input_ids, n_token_type_ids, n_attention_mask


def save_pkl(input_dict: Any, filename: str) -> None:
    """ Save pickle file
    """
    with open(f'{filename}.pkl', 'wb') as file:
        pickle.dump(input_dict, file)


def load_pkl(filepath: str) -> Any:
    """  Load pickle file
    Examples:
        filepath = './dataset_class/data_folder/train'
    """
    with open(f'{filepath}.pkl', 'rb') as file:
        output = pickle.load(file)
    return output
