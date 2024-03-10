import torch.nn as nn
from experiment.models.abstract_model import AbstractModel
from torch import Tensor
from typing import Tuple, Callable
from einops.layers.torch import Rearrange
from experiment.tuner.mlm import MLMHead
from configuration import CFG


class DistilBERT(nn.Module, AbstractModel):
    """ Main class for DistilBERT Style Model, Teacher-Student Framework
    for Knowledge Distillation aim to lighter Large Scale LLM model. This model have 3 objective functions:

        1) distillation loss, calculated by soft targets & soft predictions
            (nn.KLDIVLoss(reduction='batchmean'))

        2) student loss, calculated by hard targets & hard predictions
            (nn.CrossEntropyLoss(reduction='mean')), same as pure MLM Loss

        3) cosine similarity loss, calculated by student & teacher logit similarity
            (nn.CosineEmbeddingLoss(reduction='mean')), similar as contrastive loss

    soft targets & soft predictions are meaning that logit are passed through softmax function applied with temperature T
    temperature T aim to flatten softmax layer distribution for making "Dark Knowledge" from teacher model

    hard targets & hard predictions are meaning that logit are passed through softmax function without temperature T
    hard targets are same as just simple labels from MLM Collator returns for calculating cross entropy loss

    cosine similarity loss is calculated by cosine similarity between student & teacher
    in official repo, they mask padding tokens for calculating cosine similarity, target for this task is 1
    cosine similarity is calculated by nn.CosineSimilarity() function, values are range to [-1, 1]

    you can select any other backbone model architecture for Teacher & Student Model for knowledge distillation
    but, in original paper, BERT is used for Teacher Model & Student
    and you must select pretrained model for Teacher Model, because Teacher Model is used for knowledge distillation,
    which is containing pretrained mlm head

    Do not pass gradient backward to teacher model!!
    (teacher model must be frozen or register_buffer to model or use no_grad() context manager)

    Args:
        cfg: configuration.CFG
        model_func: make model instance in runtime from config.json

    References:
        https://arxiv.org/pdf/1910.01108.pdf
        https://github.com/huggingface/transformers/blob/main/examples/research_projects/distillation/distiller.py
    """
    def __init__(self, cfg: CFG, model_func: Callable) -> None:
        super(DistilBERT, self).__init__()
        self.cfg = cfg
        self.teacher = model_func(self.cfg.teacher_num_layers)  # must be loading pretrained model containing mlm head
        self.mlm_head = MLMHead(self.cfg)  # must be loading pretrained model's mlm head

        self.student = model_func(self.cfg.student_num_layers)
        self.s_mlm_head = MLMHead(self.cfg)

    def teacher_fw(
        self,
        inputs: Tensor,
        padding_mask: Tensor,
        attention_mask: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """ forward pass for teacher model
        """
        last_hidden_state, _ = self.teacher(
            inputs,
            padding_mask,
            attention_mask
        )
        t_logit = self.mlm_head(last_hidden_state)  # hard logit => to make soft logit
        return last_hidden_state, t_logit

    def student_fw(
        self,
        inputs: Tensor,
        padding_mask: Tensor,
        attention_mask: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """ forward pass for student model
        """
        last_hidden_state, _ = self.student(
            inputs,
            padding_mask,
            attention_mask
        )
        s_logit = self.s_mlm_head(last_hidden_state)  # hard logit => to make soft logit
        return last_hidden_state, s_logit

