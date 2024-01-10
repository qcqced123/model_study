import torch
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
        self.word_bias: Delta_E in paper
        self.abs_pos_bias: Delta_E in paper
        self.rel_pos_bias: Delta_E in paper

    References:
        https://arxiv.org/pdf/2003.10555.pdf
        https://github.com/google-research/electra
    """
    def __init__(self, cfg: CFG, model_func: Callable) -> None:
        super(ELECTRA, self).__init__()
        self.cfg = cfg
        self.generator = model_func(cfg.generator_num_layers)  # init generator
        self.mlm_head = MLMHead(self.cfg)

        self.discriminator = model_func(cfg.discriminator_num_layers)  # init generator
        self.rtd_head = RTDHead(self.cfg)

        self.share_embed_method = self.cfg.share_embed_method  # instance, es, gdes
        if self.share_embed_method == 'gdes':
            self.word_bias = nn.Parameter(
                torch.zeros(self.discriminator.embeddings.word_embeddings.weight)
            )
            self.abs_pos_bias = nn.Parameter(
                torch.zeros(self.discriminator.embeddings.abs_pos_emb.weight)
            )
            if self.cfg.model_name == 'DeBERTa':
                self.rel_pos_bias = nn.Parameter(
                    torch.zeros(self.discriminator.embeddings.rel_pos_emb.weight)
                )
        self.share_embedding()

    def share_embedding(self) -> None:
        """ init sharing options """
        if self.share_embed_method == 'instance':  # Instance Sharing
            self.discriminator.embeddings = self.generator.embeddings

        elif self.share_embed_method == 'es':  # ES (Embedding Sharing)
            self.discriminator.embeddings.word_embeddings.weight = self.generator.word_embeddings.weight
            self.discriminator.embeddings.abs_pos_emb.weight = self.generator.embeddings.abs_pos_emb.weight

            if self.cfg.model_name == 'DeBERTa':
                self.discriminator.embeddings.rel_pos_emb.weight = self.generator.embeddings.rel_pos_emb.weight

        elif self.share_embed_method == 'gdes':  # GDES (Generator Discriminator Embedding Sharing)
            self.discriminator.embeddings.word_embeddings.weight = (
                self.generator.embeddings.word_embeddings.weight.detach() + self.word_bias
            )

            self.discriminator.embeddings.abs_pos_emb.weight = (
                self.generator.embeddings.abs_pos_emb.weight.detach() + self.abs_pos_bias
            )

            if self.cfg.model_name == 'DeBERTa':
                self.discriminator.embeddings.rel_pos_emb.weight = (
                    self.generator.embeddings.rel_pos_emb.weight.detach() + self.rel_pos_bias
                )

    def generator_fw(
            self,
            inputs: Tensor,
            labels: Tensor,
            padding_mask: Tensor,
            attention_mask: Tensor = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """ forward pass for generator model
        """
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

        return g_logit, d_inputs, d_labels

    def discriminator_fw(
            self,
            inputs: Tensor,
            padding_mask: Tensor,
            attention_mask: Tensor = None
    ) -> Tensor:
        """ forward pass for discriminator model
        """
        assert inputs.ndim == 2, f'Expected (batch, sequence) got {inputs.shape}'
        d_last_hidden_states, _ = self.discriminator(
            inputs,
            padding_mask,
            attention_mask
        )
        d_logit = self.rtd_head(
            d_last_hidden_states
        )
        return d_logit
