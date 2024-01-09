import torch.nn as nn
from experiment.models.abstract_model import AbstractModel
from torch import Tensor
from typing import Tuple, Callable
from einops.layers.torch import Rearrange
from experiment.tuner.mlm import MLMHead
from experiment.tuner.rtd import get_discriminator_input, RTDHead
from configuration import CFG


class ELECTRA(nn.Module, AbstractModel):
    """ Main class for ELECTRA, having all of sub-blocks & modules such as Generator & Discriminator
    Init Scale of ELECTRA Hyper-Parameters, Embedding Layer, Encoder Blocks of Generator, Discriminator
    You can select any other backbone model architecture for Generator & Discriminator, in original paper, BERT is used

    Args:
        cfg: configuration.CFG
        model_func: make model instance in runtime from config.json

    Var:
        cfg: configuration.CFG
        generator: Generator, which is used for generating replaced tokens for RTD
                   should select backbone model ex) BERT, RoBERTa, DeBERTa, ...
        discriminator: Discriminator, which is used for detecting replaced tokens for RTD
                       should select backbone model ex) BERT, RoBERTa, DeBERTa, ...
        share_embedding: whether or not to share embedding layer (word & pos) between Generator & Discriminator

    References:
        https://arxiv.org/pdf/2003.10555.pdf
        https://github.com/google-research/electra
    """
    def __init__(self, cfg: CFG, model_func: Callable) -> None:
        super(ELECTRA, self).__init__()
        self.cfg = cfg
        self.generator = model_func()  # init generator
        self.mlm_head = MLMHead(self.cfg)

        self.discriminator = model_func()  # init generator
        self.rtd_head = RTDHead(self.cfg)

        self.share_embed = self.cfg.is_share_embed
        if self.share_embed:
            self.share_embed_method = self.cfg.share_embed_method
            self.discriminator.embeddings = self.generator.embeddings

    def forward(self, inputs: Tensor, labels: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
        assert inputs.ndim == 2, f'Expected (batch, sequence) got {inputs.shape}'
        g_last_hidden_states, _ = self.generator(
            inputs,
            padding_mask,
            attention_mask
        )
        g_logit = self.mlm_head(
            g_last_hidden_states
        )
        d_inputs, d_labels = get_discriminator_input(
            inputs,
            labels,
            g_logit,
        )
        d_last_hidden_states, _ = self.discriminator(
            d_inputs,
            padding_mask,
            attention_mask
        )
        d_logit = self.rtd_head(
            d_last_hidden_states
        )
        return g_logit, d_logit, d_labels
