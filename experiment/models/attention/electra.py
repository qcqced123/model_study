import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment.models.abstract_model import AbstractModel
from torch import Tensor
from typing import Tuple, List
from einops.layers.torch import Rearrange
from experiment.tuner.mlm import MLMHead
from experiment.tuner.rtd import post_processing, RTDHead
from deberta import DeBERTa
from configuration import CFG


class Generator(nn.Module):
    """ Backbone model with MLM Head for Generator
    Args:
        cfg: configuration.CFG
    """
    def __init__(self, cfg: CFG) -> None:
        super(Generator, self).__init__()
        self.cfg = cfg
        self.arch_name = cfg.generator
        self.generator = DeBERTa(self.cfg)
        self.mlm_head = MLMHead(self.cfg)

    def forward(self, inputs: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tensor:
        assert inputs.ndim == 2, f'Expected (batch, sequence) got {inputs.shape}'
        word_embeddings, rel_pos_emb, abs_pos_emb = self.generator.embeddings(
            inputs
        )
        last_hidden_state, hidden_states = self.generator.encoder(
            word_embeddings,
            rel_pos_emb,
            padding_mask,
            attention_mask
        )
        emd_hidden_states = hidden_states[-self.cfg.num_emd]
        emd_last_hidden_state, emd_hidden_states = self.generator.emd_encoder(
            emd_hidden_states,
            abs_pos_emb,
            rel_pos_emb,
            padding_mask,
            attention_mask
        )
        logit = self.mlm_head(emd_last_hidden_state)
        return logit


class Discriminator(nn.Module):
    """ Backbone model for Discriminator
    Args:
        cfg: configuration.CFG
    """
    def __init__(self, cfg) -> None:
        super(Discriminator, self).__init__()
        self.cfg = cfg
        self.arch_name = cfg.discriminator
        self.discriminator = DeBERTa(self.cfg)
        self.rtd_head = RTDHead(self.cfg)

    def forward(self, inputs: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        assert inputs.ndim == 2, f'Expected (batch, sequence) got {inputs.shape}'
        word_embeddings, rel_pos_emb, abs_pos_emb = self.discriminator.embeddings(
            inputs
        )
        last_hidden_state, hidden_states = self.discriminator.encoder(
            word_embeddings,
            rel_pos_emb,
            padding_mask,
            attention_mask
        )
        emd_hidden_states = hidden_states[-self.cfg.num_emd]
        emd_last_hidden_state, emd_hidden_states = self.discriminator.emd_encoder(
            emd_hidden_states,
            abs_pos_emb,
            rel_pos_emb,
            padding_mask,
            attention_mask
        )
        logit = self.rtd_head(emd_last_hidden_state)
        return logit


class ELECTRA(nn.Module, AbstractModel):
    """ Main class for ELECTRA, having all of sub-blocks & modules such as Generator & Discriminator
    Init Scale of ELECTRA Hyper-Parameters, Embedding Layer, Encoder Blocks of Generator, Discriminator
    You can select any other backbone model architecture for Generator & Discriminator, in oringinal paper, BERT is used
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
    def __init__(self, cfg: CFG) -> None:
        super(ELECTRA, self).__init__()
        self.cfg = cfg
        self.generator = Generator(self.cfg)
        self.discriminator = Discriminator(self.cfg)
        self.share_embedding = self.cfg.share_embedding
        if self.share_embedding:
            self.discriminator.discriminator.embeddings = self.generator.generator.embeddings

    def forward(self, inputs: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        assert inputs.ndim == 2, f'Expected (batch, sequence) got {inputs.shape}'
        logit = self.generator(
            inputs,
            padding_mask,
            attention_mask
        )
        post_processing(logit)