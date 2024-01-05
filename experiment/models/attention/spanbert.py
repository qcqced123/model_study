import torch.nn as nn
from experiment.models.abstract_model import AbstractModel
from torch import Tensor
from typing import Tuple, Callable
from einops.layers.torch import Rearrange
from configuration import CFG


class SpanBERT(nn.Module, AbstractModel):
    """ Main class for SpanBERT, having all of sub-blocks & modules such as self-attention, feed-forward, BERTEncoder ..
    Init Scale of SpanBERT Hyper-Parameters, Embedding Layer, Encoder Blocks

    In original paper, BERT is used as backbone model but we select DeBERTa as backbone model
    you can change backbone model to any other model easily, just passing other model name to cfg.encoder_name
    But, you must pass ONLY encoder model such as BERT, RoBERTa, DeBERTa, ...

    Args:
        cfg: configuration.CFG
        model_func: make model instance in runtime from config.json
    References:
        https://arxiv.org/pdf/1907.10529.pdf
    """
    def __init__(self, cfg: CFG, model_func: Callable) -> None:
        super(SpanBERT, self).__init__()
        self.cfg = cfg
        self.backbone = model_func

    def forward(self, inputs: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        emd_last_hidden_state, emd_hidden_states = self.backbone(
            inputs,
            padding_mask,
            attention_mask
        )
        return emd_last_hidden_state, emd_hidden_states

