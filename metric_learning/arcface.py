import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from distance_metric import SelectDistances
from typing import List, Tuple


class ArcFace(nn.Module):
    """
    ArcFace Pytorch Implementation
    Args:
        dim_model: size of hidden states(latent vector)
        num_classes: num of target classes
        s: re-scale scaler, default 30.0
        m: Additive Angular Margin Penalty
    References:
        https://github.com/wujiyang/Face_Pytorch/blob/master/margin/ArcMarginProduct.py
        https://arxiv.org/abs/1801.07698
    """
    def __init__(self, dim_model: int, num_classes: int, s: int = 30.0, m: int = 0.50) -> None:
        super(ArcFace, self).__init__()
        self.dim_model = dim_model
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.w = nn.Parameter(torch.FloatTensor(num_classes, dim_model))
        nn.init.kaiming_uniform_(self.w)  # same as nn.Linear

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - m)  # for taylor series
        self.mm = math.sin(math.pi - m) * m  # for taylor series

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        cosine = torch.matmul(F.normalize(self.w), F.normalize(inputs))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        z = cosine*self.cos_m - sine*self.sin_m
        z = torch.where(cosine > self.th, z, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        loss = (one_hot * z) + ((1.0 - one_hot) * cosine)
        loss *= self.s
        return loss

