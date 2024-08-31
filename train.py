import os
import torch
import argparse
import warnings
import trainer.train_loop as train_loop

from omegaconf import OmegaConf
from configuration import CFG
from dotenv import load_dotenv
from utils.helper import check_library, all_type_seed
from utils.util import sync_config
from huggingface_hub import login

load_dotenv()
warnings.filterwarnings('ignore')
os.environ["LRU_CACHE_CAPACITY"] = "4096"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8, expandable_segments:True"


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


def main(cfg: CFG, train_type: str, model_config: str) -> None:
    login(os.environ.get("HUGGINGFACE_API_KEY"))  # login to huggingface hub
    config_path = f'config/{train_type}/{model_config}.json'
    sync_config(cfg, OmegaConf.load(config_path))

    # init tokenizer for BPE Tokenizer
    if cfg.tokenizer.__class__.__name__ in BPE and cfg.tokenizer.pad_token is None:
        if cfg.tokenizer.bos_token.startswith('<'):
            cfg.tokenizer.add_special_tokens({'pad_token': '<pad>'})

        elif cfg.tokenizer.bos_token.startswith('['):
            cfg.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    getattr(train_loop, cfg.loop)(cfg, train_type, model_config)  # init object


if __name__ == '__main__':
    config = CFG
    parser = argparse.ArgumentParser(description="Train Script")
    parser.add_argument("train_type", type=str, help="Train Type Selection")
    parser.add_argument("model_config", type=str, help="Model config Selection")
    args = parser.parse_args()

    main(config, args.train_type, args.model_config)
