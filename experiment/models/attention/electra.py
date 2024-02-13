import torch
import torch.nn as nn
from experiment.models.abstract_model import AbstractModel
from torch import Tensor
from typing import Tuple, Callable
from einops.layers.torch import Rearrange
from experiment.tuner.mlm import MLMHead
from experiment.tuner.sbo import SBOHead
from experiment.tuner.rtd import get_discriminator_input, RTDHead
from configuration import CFG


class ELECTRA(nn.Module, AbstractModel):
    """ Main class for ELECTRA, having all of sub-blocks & modules such as Generator & Discriminator
    Init Scale of ELECTRA Hyper-Parameters, Embedding Layer, Encoder Blocks of Generator, Discriminator
    You can select any other backbone model architecture for Generator & Discriminator, in original paper, BERT is used

    if you want to use pure ELECTRA, you should set share_embedding = ES
    elif you want to use ELECTRA with GDES, you should set share_embedding = GDES
    GDES is new approach of embedding sharing method from DeBERTa-V3 paper

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
        https://arxiv.org/pdf/2111.09543.pdf
        https://github.com/google-research/electra
    """
    def __init__(self, cfg: CFG, model_func: Callable) -> None:
        super(ELECTRA, self).__init__()
        self.cfg = cfg
        self.generator = model_func(cfg.generator_num_layers)  # init generator
        self.mlm_head = MLMHead(self.cfg)
        if self.cfg.rtd_masking == 'SpanBoundaryObjective':
            self.mlm_head = SBOHead(
                cfg=self.cfg,
                is_concatenate=self.cfg.is_concatenate,
                max_span_length=self.cfg.max_span_length
            )

        self.discriminator = model_func(cfg.discriminator_num_layers)  # init generator
        self.rtd_head = RTDHead(self.cfg)

        self.share_embed_method = self.cfg.share_embed_method  # instance, es, gdes
        if self.share_embed_method == 'GDES':
            self.word_bias = nn.Parameter(
                torch.zeros_like(self.discriminator.embeddings.word_embedding.weight)
            )
            self.abs_pos_bias = nn.Parameter(
                torch.zeros_like(self.discriminator.embeddings.abs_pos_emb.weight)
            )

            delattr(self.discriminator.embeddings.word_embedding, 'weight')
            self.discriminator.embeddings.word_embedding.register_parameter('weight', self.word_bias)

            delattr(self.discriminator.embeddings.abs_pos_emb, 'weight')
            self.discriminator.embeddings.abs_pos_emb.register_parameter('weight', self.abs_pos_bias)

            if self.cfg.model_name == 'DeBERTa':
                self.rel_pos_bias = nn.Parameter(
                    torch.zeros_like(self.discriminator.embeddings.rel_pos_emb.weight)
                )
                delattr(self.discriminator.embeddings.rel_pos_emb, 'weight')
                self.discriminator.embeddings.rel_pos_emb.register_parameter('weight', self.rel_pos_emb)
        self.share_embedding()

    def share_embedding(self) -> None:
        """ init sharing options """
        def discriminator_hook(module: nn.Module, *inputs):
            if self.share_embed_method == 'instance':  # Instance Sharing
                self.discriminator.embeddings = self.generator.embeddings

            elif self.share_embed_method == 'ES':  # ES (Embedding Sharing)
                self.discriminator.embeddings.word_embedding.weight = self.generator.embeddings.word_embedding.weight
                self.discriminator.embeddings.abs_pos_emb.weight = self.generator.embeddings.abs_pos_emb.weight
                if self.cfg.model_name == 'DeBERTa':
                    self.discriminator.embeddings.rel_pos_emb.weight = self.generator.embeddings.rel_pos_emb.weight

            elif self.share_embed_method == 'GDES':  # GDES (Generator Discriminator Embedding Sharing)
                g_w_emb = self.generator.embeddings.word_embedding
                d_w_emb = self.discriminator.embeddings.word_embedding
                self._set_param(d_w_emb, 'weight', g_w_emb.weight.detach() + d_w_emb.weight)

                g_p_emb = self.generator.embeddings.abs_pos_emb
                d_p_emb = self.discriminator.embeddings.abs_pos_emb
                self._set_param(d_p_emb, 'weight', g_p_emb.weight.detach() + d_p_emb.weight)

                if self.cfg.model_name == 'DeBERTa':
                    g_rp_emb = self.generator.embeddings.rel_pos_emb
                    d_rp_emb = self.discriminator.embeddings.rel_pos_emb
                    self._set_param(d_rp_emb, 'weight', g_rp_emb.weight.detach() + d_rp_emb.weight)
        self.discriminator.register_forward_pre_hook(discriminator_hook)

    @staticmethod
    def _set_param(module, param_name, value):
        """ set param for module
        References:
             https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/apps/tasks/rtd_task.py#L132
        """
        if hasattr(module, param_name):
            delattr(module, param_name)
        module.register_buffer(param_name, value)

    def generator_fw(
            self,
            inputs: Tensor,
            labels: Tensor,
            padding_mask: Tensor,
            mask_labels: Tensor = None,
            attention_mask: Tensor = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """ forward pass for generator model
        Args:
            inputs: generator inputs
            labels: labels for generator outputs, using to make discriminator inputs, labels
            padding_mask: padding mask for inputs
            mask_labels: labels for Span Boundary Objective
            attention_mask: attention mask for inputs
        """
        assert inputs.ndim == 2, f'Expected (batch, sequence) got {inputs.shape}'
        g_last_hidden_states, _ = self.generator(
            inputs,
            padding_mask,
            attention_mask
        )
        if self.cfg.rtd_masking == 'MaskedLanguageModel':
            g_logit = self.mlm_head(
                g_last_hidden_states
            )
        elif self.cfg.rtd_masking == 'SpanBoundaryObjective':
            g_logit = self.mlm_head(
                g_last_hidden_states,
                mask_labels
            )
        pred = g_logit.clone().detach()
        d_inputs, d_labels = get_discriminator_input(
            inputs,
            labels,
            pred,
        )
        return g_logit, d_inputs, d_labels

    def discriminator_fw(
            self,
            inputs: Tensor,
            padding_mask: Tensor,
            attention_mask: Tensor = None
    ) -> Tensor:
        """ forward pass for discriminator model
        Args:
            inputs: discriminator inputs
            padding_mask: padding mask for inputs
            attention_mask: attention mask for inputs
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
