import os
import torch
import argparse
import warnings
from omegaconf import OmegaConf
from configuration import CFG
import trainer.train_loop as train_loop
from utils.helper import check_library, all_type_seed
from utils.util import sync_config
from huggingface_hub import login
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ["LRU_CACHE_CAPACITY"] = "4096"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8, expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


# tokenizer type of each model architecture
BPE = [
    'RobertaTokenizerFast',
    'GPT2TokenizerFast',
    'LlamaTokenizerFast',
]

SPM = [
    'DebertaV2TokenizerFast',
    'DebertaTokenizerFast',
    'XLMRobertaTokenizerFast',
]

WORDPIECE = [
    'BertTokenizerFast',
    'ElectraTokenizerFast',
]

check_library(True)
all_type_seed(CFG, True)
torch.cuda.empty_cache()


def main(train_type: str, model_config: str, hf_login_token: str, cfg: CFG) -> None:
    login(hf_login_token)  # login to huggingface hub
    config_path = f'config/{train_type}/{model_config}.json'
    sync_config(OmegaConf.load(config_path))

    # init tokenizer for BPE Tokenizer
    if cfg.tokenizer.__class__.__name__ in BPE and cfg.tokenizer.pad_token is None:
        if cfg.tokenizer.bos_token.startswith('<'):
            cfg.tokenizer.add_special_tokens({'pad_token': '<pad>'})

        elif cfg.tokenizer.bos_token.startswith('['):
            cfg.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    getattr(train_loop, cfg.loop)(cfg, train_type, model_config)  # init object


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Script")
    parser.add_argument("train_type", type=str, help="Train Type Selection")
    parser.add_argument("model_config", type=str, help="Model config Selection")
    parser.add_argument("hf_login_token", type=str, help="Huggingface Token")
    args = parser.parse_args()

    main(args.train_type, args.model_config, args.hf_login_token, CFG)
