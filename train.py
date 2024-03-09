import os
import torch
import argparse
import warnings
from omegaconf import OmegaConf
from configuration import CFG
import trainer.train_loop as train_loop
from utils.helper import check_library, all_type_seed
from utils.util import sync_config
from huggingface_hub import notebook_login
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LRU_CACHE_CAPACITY"] = "4096"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8, max_split_size_mb:32"
check_library(True)
all_type_seed(CFG, True)
notebook_login()  # login to huggingface hub
torch.cuda.empty_cache()


def main(train_type: str, model_config: str, cfg: CFG) -> None:
    config_path = f'config/{train_type}/{model_config}.json'
    sync_config(OmegaConf.load(config_path))  # load json config
    getattr(train_loop, cfg.loop)(cfg, train_type, model_config)  # init object


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Script")
    parser.add_argument("train_type", type=str, help="Train Type Selection")
    parser.add_argument("model_config", type=str, help="Model config Selection")
    args = parser.parse_args()

    main(args.train_type, args.model_config, CFG)
