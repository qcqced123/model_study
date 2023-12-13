import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from distance_metric import SelectDistances
from typing import Iterable, Dict


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
        """ Element-wise masking matrix generation for contrastive loss by label """
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


# Contrastive Loss for NLP Semantic Search
class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss which is basic method of Metric Learning
    Closer distance between data points in intra-class, more longer distance between data points in inter-class
    Distance:
        Euclidean Distance: sqrt(sum((x1 - x2)**2))
        Cosine Distance: 1 - torch.nn.function.cos_sim(x1, x2)
    Examples:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        train_examples = [
            InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
            InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0)]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
        train_loss = losses.ContrastiveLoss(model=model)
    Args:
        margin: margin value meaning for Area of intra class(positive area), default 1.0
        metric: standard of distance metrics, default cosine distance
    References:
        https://github.com/KevinMusgrave/pytorch-metric-learning
        https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/ContrastiveLoss.py
        https://www.youtube.com/watch?v=u-X_nZRsn5M&list=LL&index=3&t=10s&ab_channel=DeepFindr
    """
    def __init__(self, metric: str = 'cosine', margin: int = 1.0) -> None:
        super(ContrastiveLoss, self).__init__()
        self.distance = SelectDistances(metric)  # Options: euclidean, manhattan, cosine
        self.margin = margin

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor) -> Tensor:
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        anchor, instance = embeddings
        distance = self.distance(anchor, instance)
        contrastive_loss = 0.5 * (labels.float() * torch.pow(distance, 2) +
                                  (1 - labels.float()) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return contrastive_loss.mean()
