import torch.nn as nn

from torch import Tensor


def average_sequence_embedding(x: Tensor, mask: Tensor) -> Tensor:
    """ function for calculating the average of sequence embedding with mask
    for making inputs to Deep Averaging Network
    """
    x = x * mask
    valid_counts = mask.sum()  # number of valid tokens in each sequence
    return x.mean(dim=1) / valid_counts


class FeedForward(nn.Module):
    """ class for Feed-Forward Network module in DAN (Deep Averaging Network)
    in dan, feedforward network is used as one layer or encoder in whole architecture

    Also, we use GELU activation function for hidden layer, and dropout for regularization
    but, in original paper, they do not use any activation function, dropout for feedforward layer,

    Args:
        dim_model: dimension of model's latent vector space, default 1024
        dim_ffn: dimension of FFN's hidden layer, default 4096 from official paper
        hidden_dropout_prob: dropout rate, default 0.1

    Math:
        FeedForward(x) = FeedForward(LN(x))+x
    """
    def __init__(self, dim_model: int = 1024, dim_ffn: int = 4096, hidden_dropout_prob: float = 0.1) -> None:
        super(FeedForward, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_ffn),
            nn.GELU(),
            nn.Dropout(p=hidden_dropout_prob),
            nn.Linear(dim_ffn, dim_model),
            nn.Dropout(p=hidden_dropout_prob),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)


class DeepAveragingNetwork(nn.Module):
    """ module for Deep Averaging Network (DAN), which is version of Bag of Words (BoW) model with deep learning

    implementations:
        1) averaging the input sequence embeddings
        2) pass the averaged embeddings to the N stacked feedforward network
        3) get logit and probability to the target classes

    Args:
        num_layers: number of layers in DAN, default 4
        dim_model: dimension of model's latent vector space, default 768 dimension
        dim_ffn: dimension of FFN's hidden layer, default 3072 dimension
        hidden_dropout_prob: dropout rate, default 0.1

    Reference:
        https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf
    """
    def __init__(
        self,
        num_layers: int = 4,
        dim_model: int = 768,
        dim_ffn: int = 3072,
        hidden_dropout_prob: float = 0.1
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dim_model = dim_model
        self.dim_ffn = dim_ffn
        self.hidden_dropout_prob = hidden_dropout_prob
        self.averaging_emb = average_sequence_embedding
        self.layer = nn.ModuleList(
            [FeedForward(self.dim_model, self.dim_ffn, self.hidden_dropout_prob) for _ in range(self.num_layers)]
        )

    def forward(self, x: Tensor, padding_mask: Tensor = None) -> Tensor:
        v = self.averaging_emb(
            x,
            padding_mask
        )

        for layer in self.layer:
            v = layer(v)

        return v
