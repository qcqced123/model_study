""" py module of MoE (Mixture of Experts) implementation for transformer architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple, List
from einops.layers.torch import Rearrange
from transformers import AutoConfig, AutoTokenizer
from experiment.models.abstract_model import AbstractModel


class Experts(nn.Module):
    """ replacing the pure feed-forward network for reducing the computational cost, time complexity
    Args:
        dim_model (int): size of latent space of current architecture model
    """
    def __init__(self, dim_model: int):
        super(Experts, self).__init__()
        self.expert = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.ReLU(),
            nn.Linear(dim_model, dim_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.expert(x)


class SparseMoELayer(nn.Module):
    """ main module of MoE, replacing the dense feed-forward network in transformer encoder/decoder blocks.
    this module has the gating network, experts network

    sparse MoE layer is originated from switch transformer.

    Args:
        k (int): value of top-k routing
        dim_model(int): value of gating network's input size

    Design:
        - single(sparse) expert routing
        - minimum contribution masking algorithm: select masking tokens, when routing result would be overflowed
        - gather-scatter method for routing to activate expert

    Reference:
        - https://huggingface.co/blog/moe
        - https://arxiv.org/pdf/2101.03961  # switch transformer paper
        - https://pytorch.org/docs/stable/generated/torch.full.html
        - https://pytorch.org/docs/main/generated/torch.nonzero.html
        - https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html
    """
    def __init__(self, num_experts: int, dim_model: int, k: int = 1, capacity_factor: float = 1.25):
        super(SparseMoELayer, self).__init__()
        self.k = k  # value of top-k routing
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.gate = nn.Sequential(
            nn.Linear(dim_model, num_experts),
            nn.Softmax(dim=-1)
        )
        self.experts = nn.ModuleList([Experts(dim_model) for _ in range(self.num_experts)])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # check input data's validation 2
        # assert x.ndim != 3, f'Expected (batch, sequence, hidden_state) got {x.shape}'

        # aliasing the each tensor dimension
        # calculating the expert capacity value
        batch_size, seq_len, dim_model = x.shape
        total = batch_size * seq_len
        capacity = int((total/self.num_experts) * self.capacity_factor)

        # single (top-1) expert routing
        # gating method must have the softmax layer
        # 1) set the token limit of each unique expert
        # 2) forward the flatten input x into gating layer
        # 3) get top-1 expert's name for sparse routing
        # 4) make the one-hot vector by num_experts
        gating_score = self.gate(x)
        expert_indices = gating_score.argmax(dim=-1)

        flat_x = x.view(total, dim_model)
        token_mask = torch.full((total, ), fill_value=-1, dtype=torch.long, device=x.device)
        for expert in range(self.num_experts):
            indices = (expert == expert_indices).nonzero(as_tuple=False).squeeze()
            if not indices.numel():
                continue

            if not indices.dim():
                indices = indices.unsqueeze(0)

            # exception handling: when the allocated tokens are over-flowed to expert capacity
            # current masking algorithm is "random sampling"
            # must add the alternative method, named "minimum contribution"
            if indices.numel() > capacity:  # torch.numel(): count the element in tensor
                indices = indices[:capacity]

            token_mask[indices] = expert

        # gathering the each tokens, breakdown by expert name
        # scattering the gather tokens into original sequence ordering
        output = torch.zeros_like(flat_x)
        for expert in range(self.num_experts):
            indices = (expert == token_mask).nonzero(as_tuple=False).squeeze()
            if not indices.numel():
                continue

            expert_input = flat_x[indices]
            output[indices] = self.experts[expert](expert_input)

        output = output.view(batch_size, seq_len, dim_model)

        # calculate the load balancing loss for making the gating layer to distribute the score uniformly
        # element-wise product between "expert probs" and "expert density"
        # expert probs: average gating scores of each expert
        # expert density: average fraction of tokens routed to each expert
        expert_probs = gating_score.view(total, self.num_experts).mean(dim=0)
        expert_density = token_mask[token_mask > -1].bincount() / (total - len(token_mask[token_mask == -1]))

        expert_losses = (expert_probs * expert_density) * self.num_experts ** 2
        expert_loss = expert_losses.mean()

        return output, expert_loss


if __name__ == '__main__':
    # set the accelerator device
    os = "macOS"
    accelerator = None
    if os == "macOS":
        accelerator = "mps" if torch.backends.mps.is_available() else "cpu"
    elif os == "linux":
        accelerator = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    device = torch.device(accelerator)

    # init the sparse MoE module
    num_experts = 16
    batch_size = 16
    max_seq = 512
    dim_model = 512
    sparse_moe = SparseMoELayer(
        num_experts=num_experts,
        dim_model=dim_model
    )
    sparse_moe.to(device=device)
    # check the module of sparse MoE
    print(sparse_moe)

    # forward the input x to sparse MoE to debugging
    x = torch.randn(batch_size, max_seq, dim_model, device=device)
    y, l = sparse_moe(x)
    print(f"final output activation of sparse MoE layer is: {y}")
    print(f"final load balancing loss of sparse MoE layer is: {l}")
