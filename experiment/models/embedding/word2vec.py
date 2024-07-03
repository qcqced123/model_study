import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple, List


class Word2Vec(nn.Module):
    """ module for word2vec model from google research 2013, especially implemented for skip-gram model

    Architecture:
        1) build the negative sample for skip-gram word2vec
            - calculate the probability of negative sampling (unigram distribution)

    Args:

    References:

    """
    def __init__(self, cfg) -> None:
        super(Word2Vec, self).__init__()
        self.cfg = cfg
        self.prob_sampling = self.get_prob_sampling
        self.classifier = nn.Linear(self.cfg.hidden_size, 2)  # for binary classification (out-of-context, in-context)

    def get_prob_sampling(self, inputs_ids: Tensor = None) -> Tensor:
        """ function for getting the probability of negative sampling, sub-sampling in word2vec skip-gram model

        Math:
            Pn(w) = U(w)^3/4 / sum(U(w)^3/4)  (U: unigram distribution)
        """
        pow_value = self.cfg.pow_value

    def build_negative_sampling(self):
        pass

    def forward(self, x: Tensor) -> Tensor:
        pass






