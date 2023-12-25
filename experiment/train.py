import argparse
import os, warnings
from omegaconf import OmegaConf

from configuration import CFG
import trainer.train_loop as train_loop
from utils.helper import check_library, all_type_seed
from utils.util import sync_config
from dataset_class.preprocessing import add_target_token, add_anchor_token
from huggingface_hub import notebook_login
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LRU_CACHE_CAPACITY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.9,max_split_size_mb:512"
check_library(True)
all_type_seed(CFG, True)
notebook_login()  # login to huggingface hub


def main(config_path: str, cfg: CFG) -> None:
    target_token, anchor_token = ' [TAR] ', ' [ANC] '
    sync_config(OmegaConf.load(config_path))  # load json config
    add_target_token(cfg, target_token), add_anchor_token(cfg, anchor_token)
    # cfg = OmegaConf.structured(CFG)
    # OmegaConf.merge(cfg)  # merge with cli_options
    getattr(train_loop, cfg.loop)(cfg)  # init object


if __name__ == '__main__':
    main('deberta_cfg.json', CFG)

