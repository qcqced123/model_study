import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from distance_metric import SelectDistances


# Batch Embedding Table Dot-Product Version of Contrastive Loss
class BatchDotProductContrastiveLoss(nn.Module):
    """
    Batch Embedding Inner-Product based Contrastive Loss Function
    This metric request just one input batch, not two input batches for calculating loss
    Args:
        metric: distance metrics, default: cosine
        margin: margin for negative pairs, default: 1.0
    Maths:
        loss(x, y) = 0.5 * (y * distance(x1, x2) + (1 - y) * max(margin - distance(x1, x2), 0))
    """
    def __init__(self, metric: str = 'cosine', margin: int = 1.0) -> None:
        super(BatchDotProductContrastiveLoss, self).__init__()
        self.distance = SelectDistances(metric)  # Options: euclidean, manhattan, cosine
        self.margin = margin

    @staticmethod
    def elementwise_labels(y: torch.Tensor) -> torch.Tensor:
        f_mask = (y == 1) | (y == -1)
        t_mask = (y == 2) | (y == 0)
        y = y.unsqueeze(1)
        label = y + y.T
        label[f_mask], label[t_mask] = -1, 1
        label[f_mask] = 0
        label = torch.triu(label, diagonal=1)
        return label

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = self.elementwise_labels(labels)
        distance = self.distance.select_distance(emb, emb)
        contrastive_loss = 0.5 * (labels.float() * torch.pow(distance, 2) +
                                  (1 - labels.float()) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        contrastive_loss = torch.sum(torch.triu(contrastive_loss, diagonal=1))
        return contrastive_loss / labels.shape[0]
