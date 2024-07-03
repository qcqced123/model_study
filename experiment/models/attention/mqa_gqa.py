"""Python Module for implementing multi-query attention, grouped multi-query attention (MQA, GQA)
You can choose the head types as 'mha', 'mqa', 'gqa' for the attention head type
GQA is the grouped multi-query attention, which is the optimized version of the multi-query attention
applying to Llama2, Phi3 ... (modern LLM)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class AttentionHeads(nn.Module):
    """ module for implementing original multi-head attention, multi-query attention, grouped multi-query attention (MQA, GQA)

    hidden state dimension(last dim of each tensor) of the
    key, value tensor must be set to the same value as single head's hidden state dimension(=self.dim_heads in source code)

    if you set the head_types as 'mha', the model will be implemented as original multi-head attention head
    if you set the head_types as 'mqa', the model will be implemented as multi-query attention head: head number is just one
    if you set the head_types as 'gqa', the model will be implemented as grouped multi-query attention head: head number is divided by N // 8

    Args:
        cfg: configuration module for setting the model hyperparameters

    References:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/pdf/1911.02150
        https://arxiv.org/pdf/2305.13245
        https://github.com/fkodom/grouped-query-attention-pytorch/blob/main/grouped_query_attention_pytorch/attention.py
    """

    def __init__(self, cfg) -> None:
        super(AttentionHeads, self).__init__()
        self.head_types = cfg.head_types
        self.dim_model = cfg.dim_model
        self.q_heads = cfg.q_heads
        self.kv_heads = cfg.q_heads
        self.dim_heads = self.dim_model // self.q_heads

        # set the dimension size of the projection matrix
        self.dim_proj = None
        if self.head_types == "mha":
            self.dim_proj = self.dim_model

        elif self.head_types == "mqa":
            self.dim_proj = self.dim_heads

        elif self.head_types == "gqa":
            self.kv_heads = self.q_heads // 8
            self.dim_proj = self.dim_heads

        self.proj_q = nn.Linear(self.dim_model, self.dim_model)
        self.proj_k = nn.Linear(self.dim_model, self.dim_proj)
        self.proj_v = nn.Linear(self.dim_model, self.dim_proj)

        self.dot_scale = torch.sqrt(torch.tensor(self.dim_heads))
        self.dropout = nn.Dropout(0.1)

        self.fc_o = nn.Linear(self.dim_model, self.dim_model)

    def get_proj_matrix(self, x: Tensor, projector: nn.Linear) -> Tensor:
        # aliasing the tensor dimension
        y = None
        batch_size, seq_len, _ = x.shape
        if self.head_types == "mha":
            y = projector(x).reshape(
                batch_size,
                seq_len,
                self.q_heads,
                self.dim_heads
            ).permute(0, 2, 1, 3).contiguous()

        elif self.head_types == "mqa":
            y = projector(x).reshape(
                batch_size,
                seq_len,
                self.dim_proj
            ).unsqueeze(1)

        elif self.head_types == "gqa":
            y = projector(x).reshape(
                batch_size,
                seq_len,
                self.dim_proj
            ).unsqueeze(1).expand(-1, self.kv_heads, -1, -1)
        return y

    def scaled_dot_product_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        dot_scale: Tensor,
        attention_dropout: nn.Dropout,
        padding_mask: Tensor = None,
        attention_mask: Tensor = None
    ) -> Tensor:
        """ Scaled Dot-Product attention with Masking for padding mask, parallel version for Multi-Head Attention

        Args:
            q: query matrix, shape (batch_size, seq_len, dim_head)
            k: key matrix, shape (batch_size, seq_len, dim_head)
            v: value matrix, shape (batch_size, seq_len, dim_head)
            dot_scale: scale factor for Q•K^T result
            attention_dropout: dropout for attention matrix, default rate is 0.1 from official paper
            padding_mask: mask for attention matrix for MLM, you must check whether or not padding token is 1
            attention_mask: mask for attention matrix for CLM

        Math:
            A = softmax(q•k^t/sqrt(D_h)), SA(z) = Av

        Reference:
            https://arxiv.org/abs/1706.03762
            https://arxiv.org/pdf/1810.04805.pdf
        """

        if self.head_types == "gqa":
            batch_size, q_heads, seq_len, dim_proj = q.shape
            q = q.reshape(batch_size, q_heads // self.kv_heads, self.kv_heads, seq_len, dim_proj)
            k = k.unsqueeze(1)

        attention_matrix = torch.matmul(q, k.transpose(-1, -2)) / dot_scale
        if self.head_types == "gqa":
            attention_matrix = attention_matrix.reshape(batch_size, q_heads, seq_len, seq_len)

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # for broadcasting: shape (BS, 1, 1, SEQ_LEN)
            attention_matrix = attention_matrix.masked_fill(padding_mask == 1, float('-inf'))

        attention_dist = attention_dropout(
            F.softmax(attention_matrix, dim=-1)
        )

        # aliasing the tensor dimension
        _, q_heads, _, _ = attention_dist.shape
        batch_size, _, seq_len, dim_proj = v.shape

        attention_score = None
        if self.head_types == "gqa":
            attention_dist = attention_dist.reshape(
                batch_size,
                q_heads // self.kv_heads,
                self.kv_heads,
                seq_len,
                seq_len
            )
            v = v.unsqueeze(1)
            attention_score = torch.matmul(attention_dist, v).reshape(batch_size, q_heads, seq_len, dim_proj).permute(0,
                                                                                                                      2,
                                                                                                                      1,
                                                                                                                      3).reshape(
                batch_size,
                seq_len,
                q_heads * dim_proj,
            ).contiguous()

        else:
            attention_score = torch.matmul(attention_dist, v).permute(0, 2, 1, 3).reshape(
                batch_size,
                seq_len,
                q_heads * dim_proj
            ).contiguous()

        return attention_score

    def forward(self, x: Tensor) -> Tensor:
        # need to set assert code line in here
        # aliasing the tensor dimension
        batch_size, seq_len, _ = x.shape
        q = self.proj_q(x).reshape(
            batch_size,
            seq_len,
            self.q_heads,
            self.dim_heads
        ).permute(0, 2, 1, 3).contiguous()
        k = self.get_proj_matrix(x, self.proj_k)
        v = self.get_proj_matrix(x, self.proj_v)

        attention_score = self.scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            dot_scale=self.dot_scale,
            attention_dropout=self.dropout
        )
        attention_output = self.fc_o(attention_score)
        return attention_output


class TestConfig:
    head_types = "gqa"
    dim_model = 1024
    max_len = 512
    q_heads = 16
    batch_size = 16


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = TestConfig()
    x = torch.randn(cfg.batch_size, cfg.max_len, cfg.dim_model).to(device)
    mha = AttentionHeads(cfg).to(device)
    output = mha(x)
    print(f"current attention output shape: {output.shape}")