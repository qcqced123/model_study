import os
import warnings
from omegaconf import OmegaConf
from configuration import CFG
import trainer.train_loop as train_loop
from utils.helper import check_library, all_type_seed
from utils.util import sync_config
from huggingface_hub import notebook_login

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LRU_CACHE_CAPACITY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.95, max_split_size_mb:512"

check_library(True)
all_type_seed(CFG, True)
notebook_login()  # login to huggingface hub


def main(config_path: str, cfg: CFG) -> None:
    sync_config(OmegaConf.load(config_path))  # load json config
    # cfg = OmegaConf.structured(CFG)
    # OmegaConf.merge(cfg)  # merge with cli_options
    getattr(train_loop, cfg.loop)(cfg)  # init object


if __name__ == '__main__':
    main('config/bert_cfg.json', CFG)

