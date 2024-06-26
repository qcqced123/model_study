"""ASAP, we must select the dot_scale for the attention module, because the dot_scale is the important hyperparameter
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from configuration import CFG
from typing import Tuple, List
from experiment.models.abstract_model import AbstractModel


def make_global_attention_indices(
    input_ids: Tensor,
    global_attention_tokens_id_list: List[int]
) -> Tuple[Tensor, Tensor]:
    """ function for building the global attention indexes tensors

    implementations:
        1) find the indices of the special tokens
        2) make the batch-wise indices tensor
        3) make the indices tensor for indexing batch-dimension in query, key, value, attention matrix

    References:
        https://pytorch.org/docs/stable/generated/torch.argwhere.html#torch.argwhere
    """

    # find the indices of the special tokens
    batch_size, _ = input_ids.shape
    global_attention_tokens_mask = torch.isin(
        input_ids, torch.tensor(global_attention_tokens_id_list)
    )
    global_attention_tokens_indices = torch.argwhere(global_attention_tokens_mask)

    # make the batch-wise indices tensor
    unique_elements, counts = torch.unique(global_attention_tokens_indices[:, 0], return_counts=True)
    reshaped = global_attention_tokens_indices[:, 1].reshape(len(unique_elements), -1)

    # make the indices tensor for indexing batch-dimension in query, key, value, attention matrix
    row_indices = torch.arange(batch_size).unsqueeze(1).to(global_attention_tokens_indices.device)
    return row_indices, reshaped


def global_attention(
    query: Tensor,
    key: Tensor,
    row_indices: Tensor,
    global_attention_tokens_indices_padded: Tensor
) -> Tensor:
    """ function for global full attention, applying selected index of token, meaning of task-specific tokens,
    custom user added tokens such as [CLS], [SEP], [ANSWER], [QUESTION] ...

    this function selects the global attention tokens from the key tensor
    and then do the dot-product between query tensor and selected key tensor

    Args:
        query: torch.Tensor with shape [batch, seq_len, heads, dim_head]
        key: torch.Tensor with shape [batch, seq_len, heads, dim_head]
        row_indices: torch.Tensor with shape [batch], meaning for token indices from torch.arange(batch_size)
        global_attention_tokens_indices_padded: torch.Tensor with shape [batch, nums_global_attention_tokens]

    Returns:
        attention matrix: dot-product result with shape [batch, seq_len, heads, nums_global_attention_tokens]
    """
    selected_k = key[row_indices, global_attention_tokens_indices_padded, ...].squeeze(2).transpose(2, 1)
    attention_matrix = torch.matmul(query.transpose(2, 1), selected_k.transpose(-2, -1))
    return attention_matrix


def sliding_window_attention(
    query: Tensor,
    key: Tensor,
    window_size: int,
    window_overlap: int
) -> Tensor:
    """ function for sliding window attention from longformer,
    this is the customized version, original source code from huggingface.transformers.model.longformer.modeling_longformer.py

    implementation:
        1) make the overlapping chunks of each tensor (query, key)
        2) apply the sliding window attention (same as convolution attention, local context window attention)
        3) apply the attention, padding mask for chunked attention matrix

    Args:
        query: torch.Tensor with shape [batch, seq_len, heads, dim_head]
        key: torch.Tensor with shape [batch, seq_len, heads, dim_head]
        window_size: int, the size of local context window
        window_overlap: int, the overlap size between two consecutive windows, same as half of the window_size

    References:
        https://github.com/allenai/longformer/blob/master/longformer/longformer.py
        https://github.com/allenai/longformer/blob/master/longformer/sliding_chunks.py
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/longformer/modeling_longformer.py
    """
    # aliasing the tensor dimensions
    # number of chunks in current batches
    batch_size, seq_len, num_heads, dim_head = query.size()
    chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1

    # integrate the batch size and num_heads dimensions into the seq_len dimension
    query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, dim_head)
    key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, dim_head)

    # make the over-lapping chunks of each tensor
    chunked_q = chunk_emb(
        hidden_states=query,
        window_size=window_size,
    )
    chunked_k = chunk_emb(
        hidden_states=key,
        window_size=window_size,
    )

    # result of sliding window query-key dot-product
    # bcxd: batch*heads, chunk, query_window_size, dim_head
    # bcyd: batch*heads, chunk, key_window_size, dim_head
    # test_chunked_attention_scores = torch.matmul(chunked_q, chunked_k.transpose(-2, -1))
    diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (chunked_q, chunked_k))

    # convert diagonals into columns
    # tensor shape: [batch*heads*chunks, dim_head, window_size+1]
    # add just one padding value into the window_size dimension
    diagonal_chunked_attention_scores = pad_and_transpose_last_two_dims(
        diagonal_chunked_attention_scores,
        padding=(0, 0, 0, 1)
    )

    # allocate memory space for the overall attention matrix where the chunks are combined
    # the last dimension is window_size + 1
    # for the making sure the allocate the new tensor on the same device with the input tensor,
    # use the input_tensor.new_zeros() function, it will allocate the new tensor on the same device
    diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        (batch_size * num_heads, chunks_count + 1, window_overlap, window_size + 1)
    )
    diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[:, :, :window_overlap, :window_overlap + 1]
    diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[:, -1, window_overlap:, :window_overlap + 1]
    diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[:, :, -(window_overlap + 1):-1, window_overlap + 1:]
    diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[:, 0, :window_overlap - 1, 1 - window_overlap:]

    # tensor shape: batch, seq_len, heads, window_size+1
    # not yet recover the original shape of the attention matrix
    diagonal_attention_scores = diagonal_attention_scores.view(
        batch_size,
        num_heads,
        seq_len,
        window_size + 1
    ).transpose(2, 1)

    _mask_invalid_locations(diagonal_attention_scores, window_overlap)
    return diagonal_attention_scores


def chunk_emb(
    hidden_states: torch.Tensor,
    window_size: int
) -> torch.Tensor:
    """ convert full tensors into overlapping chunks tensors,
    you must pass the dividable size of sequence length and the window size

    this is the customized version, original source code from huggingface.transformers.model.longformer.modeling_longformer.py

    implementations:
        1) make the non-overlapping chunks
            - split the seq_len dimension into num_chunks an context_window size

        2) get the tensor-storage meta data for using torch.as_strided() function for sliding window attention
            - change the meta data for the overlapping chunks

        3) apply as_strided function for making overlapping chunks

    Args:
        hidden_states: torch.Tensor with shape [batch*heads, seq_len, dim_head]
        window_size: int, the size of local context window

    References:
        https://github.com/allenai/longformer/blob/master/longformer/longformer.py
        https://github.com/allenai/longformer/blob/master/longformer/sliding_chunks.py
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/longformer/modeling_longformer.py
    """
    # aliasing the tensor dimensions
    # make non-overlapping chunks
    bs_heads, seq_len, dim_head = hidden_states.size()

    h = hidden_states.view(
        bs_heads,
        torch.div(seq_len, window_size, rounding_mode="trunc"),
        window_size,
        dim_head
    )
    # use `torch.as_strided()` to make the chunks overlap with an overlap size = window_overlap
    # make overlapping chunks, so number of chunks will be doubled, and stride of the num_chunks dim will be halved
    chunk_size = list(h.size())  # batch_size*num_heads, chunks, window_size, dim_head
    chunk_size[1] = chunk_size[1] * 2 - 1

    chunk_stride = list(h.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)  # for sliding window


def pad_and_transpose_last_two_dims(hidden_states_padded: torch.Tensor, padding: Tuple) -> torch.Tensor:
    """
    Args:
        hidden_states_padded: torch.Tensor with shape [batch*heads, chunks, window_size, dim_head]
        padding: Tuple, padding value for padding the hidden_states_padded tensor

    References:
        https://github.com/allenai/longformer/blob/master/longformer/longformer.py
        https://github.com/allenai/longformer/blob/master/longformer/sliding_chunks.py
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/longformer/modeling_longformer.py
    """
    # why add the padding value into hidden dimension of the attention heads..?
    # padding value is not important because it will be overwritten
    h_padded = nn.functional.pad(hidden_states_padded, padding)
    bs_heads, chunks, window_size, dim_head = h_padded.size()
    h_padded = h_padded.view(
        bs_heads,
        chunks,
        dim_head,
        window_size
    )
    return h_padded


def _mask_invalid_locations(input_tensor: torch.Tensor, affected_seq_len: int) -> None:
    """ mask the invalid locations in the attention matrix

    Args:
        input_tensor: torch.Tensor with shape [batch, seq_len, heads, window_size+1]
        affected_seq_len: int, the size of sliding window (local context window, convolution window)

    References:
        https://github.com/allenai/longformer/blob/master/longformer/longformer.py
        https://github.com/allenai/longformer/blob/master/longformer/sliding_chunks.py
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/longformer/modeling_longformer.py
    """
    beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    beginning_mask = beginning_mask_2d[None, :, None, :]  # add the new dimension for the batch size and heads
    ending_mask = beginning_mask.flip(
        dims=(1, 3))  # flip the tensor by sequence length dimension, window size dimension
    beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_mask = beginning_mask.expand(beginning_input.size())  # expand the mask tensor

    # make the new tensor with the same shape of the input tensor and fill the tensor with -inf
    # and then using where function for applying the mask tensor, only masking index element will be converted to -inf
    # if mask value is True, the original value will be assigned to the tensor
    # elif mask value is False, the tensor will be filled with -inf
    input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        beginning_input, -float("inf")
    ).where(beginning_mask.bool(), beginning_input)

    ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
    ending_mask = ending_mask.expand(ending_input.size())
    input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):] = torch.full_like(
        ending_input, -float("inf")
    ).where(ending_mask.bool(), ending_input)

    ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
    ending_mask = ending_mask.expand(ending_input.size())
    input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):] = torch.full_like(
        ending_input, -float("inf")
    ).where(ending_mask.bool(), ending_input)


def dilated_sliding_window_attention():
    """ dilated sliding window attention need to use custom CUDA kernel from original paper
    so, we don't implement this function, just use two functions above (global, sliding window)
    """
    pass


def compute_attn_output_with_global_indices(
    value_vectors: Tensor,
    attention_dist: Tensor,
    row_indices: Tensor,
    max_num_global_attn_indices: int,
    global_attention_tokens_indices: Tensor,
    one_sided_attn_window_size: int
):
    """ function for computing the global attention output and local attention output independently,
    and then add the two outputs, recovering the original shape of the attention output (batch, seq_len, heads, dim_head)

    this is the customized version, original source code from huggingface.transformers.model.longformer.modeling_longformer.py

    output tensor of local and global attention must have the same shape: [batch, seq_len, heads, dim_head]

    value vectors shape: [batch, seq_len, heads, dim_head]
    attention_dist shape: [batch, seq_len, num_heads, num_global_attn_tokens + window_size + 1]

    implementations:
        1) compute only global attention output
        2) compute only local attention output
        3) add the global and local attention output

    Args:
        value_vectors: torch.Tensor with shape [batch, num_heads, seq_len, dim_head]
        attention_dist: torch.Tensor with shape [batch, seq_len, num_heads, num_global_attn_tokens + window_size + 1]
        row_indices: torch.Tensor, the indices of the batch size dimension
        max_num_global_attn_indices: int, the number of global attention tokens

        global_attention_tokens_indices: ??
        one_sided_attn_window_size: int, the size of local context window
    """
    # cut local attn probs to global only for calculating global attn output
    # torch.narrow: view function of tensor, memory address is same
    attn_dist_only_global = attention_dist.narrow(-1, 0, max_num_global_attn_indices)

    # get value vectors for global only
    # make the empty tensor for selected global value vectors
    # fill the tensor to each index of global attention indices
    value_vectors_only_global = value_vectors[row_indices, global_attention_tokens_indices, ...].squeeze(2).transpose(2,1)

    # compute attn output only global
    # use `matmul` because `einsum` crashes sometimes with fp16
    # torch.matmul([seq_len, global_tokens], [global_tokens, dim_heads])
    # output shape: [batch, seq_len, heads, dim_head]
    attn_output_only_global = torch.matmul(
        attn_dist_only_global.clone(),
        value_vectors_only_global.clone()
    ).transpose(1, 2)

    # reshape attn dist
    # for computing local attention output
    attn_dist_without_global = attention_dist.narrow(
        -1, max_num_global_attn_indices, attention_dist.size(-1) - max_num_global_attn_indices
    ).contiguous()

    # compute the local attention output
    # return tensor shape must be same as global attention output: batch, seq_len, heads, dim_head
    attn_output_without_global = sliding_chunks_matmul_attn_probs_value(
        attn_dist_without_global,
        value_vectors,
        one_sided_attn_window_size  # same as window_overlap
    )
    return attn_output_only_global + attn_output_without_global


def sliding_chunks_matmul_attn_probs_value(
    attention_dist: Tensor,
    value: Tensor,
    window_overlap: int
):
    """ same role as sliding_chunks_query_key_matmul but for attention_dist and value tensors.
    return tensor will be of the same shape as `attention_dist`.

    this is the customized version, original source code from huggingface.transformers.model.longformer.modeling_longformer.py

    Args:
        attention_dist: torch.Tensor with shape [batch, seq_len, num_heads, window_size+1]
        value: torch.Tensor with shape [batch, seq_len, num_heads, dim_head]
        window_overlap: int, the overlap size between two consecutive windows, same as half of the window_size
    """
    batch_size, seq_len, num_heads, head_dim = value.size()
    chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1

    # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap
    chunked_attn_probs = attention_dist.transpose(1, 2).reshape(
        batch_size * num_heads,
        torch.div(seq_len, window_overlap, rounding_mode="trunc"),
        window_overlap,
        2 * window_overlap + 1,
    )

    # group batch_size and num_heads dimensions into one
    value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

    # pad seq_len with w at the beginning of the sequence and another window overlap at the end
    padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

    # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
    chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
    chunked_value_stride = padded_value.stride()
    chunked_value_stride = (
        chunked_value_stride[0],
        window_overlap * chunked_value_stride[1],
        chunked_value_stride[1],
        chunked_value_stride[2],
    )
    chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

    chunked_attn_probs = _pad_and_diagonalize(chunked_attn_probs)

    # calculate the local attention output
    # chunked_attn_probs shape: total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
    context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)


def _pad_and_diagonalize(chunked_hidden_states: Tensor) -> Tensor:
    """ shift every row 1 step right, converting columns into diagonals.
    original source code from huggingface transformers.models.longformer.modeling_longformer.py

    Args:
        chunked_hidden_states: torch.Tensor with shape [total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim]
    """
    # aliasing to tensor dimensions
    # padding to second dimension: num_chunks
    # add window_overlap + 1 size of padding into "num_chunks" dimension
    total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states.size()

    chunked_hidden_states = nn.functional.pad(
        chunked_hidden_states, (0, window_overlap + 1)
    )

    chunked_hidden_states = chunked_hidden_states.view(
        total_num_heads, num_chunks, -1
    )
    # remove padding from the last dimension
    # total_num_heads x num_chunks x window_overlap*window_overlap
    chunked_hidden_states = chunked_hidden_states[:, :, :-window_overlap]
    chunked_hidden_states = chunked_hidden_states.view(
        total_num_heads,
        num_chunks,
        window_overlap,
        window_overlap + hidden_dim
    )
    chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    return chunked_hidden_states


def longformer_attention(
    local_q: Tensor,
    local_k: Tensor,
    local_v: Tensor,
    global_q: Tensor,
    global_k: Tensor,
    global_v: Tensor,
    row_indices: Tensor,
    global_attention_tokens_indices_padded: Tensor,
    window_size: int,
    window_overlap: int,
    attention_mask: Tensor = None,
    attention_dropout: nn.Dropout = 0.1
) -> Tensor:
    """ function for the longformer total attention (sliding window attention + global full attention)
    this is the customized version, original source code from huggingface.transformers.model.longformer.modeling_longformer.py

    implementations:
        1) sliding window attention
        2) dilated sliding window attention (not supported in general CUDA)
        3) global full attention with task-specific tokens
        4) concatenate the sliding window attention matrix and global attention matrix
        5) apply the softmax function into row-wise of the attention matrix
        6) calculate the local attention output and global attention output independently
        7) recover the original shape of the attention output: [batch, seq_len, heads*dim_head(=dim_model)]

    Args:
        all of the input tensors have the same shape [batch, seq_len, heads, dim_head]

        local_q: projected query tensor from local query projector
        local_k: projected key tensor from local key projector
        local_v: projected value tensor from local value projector

        global_q: projected query tensor from global query projector
        global_k: projected key tensor from global key projector
        global_v: projected value tensor from global value projector
        row_indices: torch.Tensor, the indices of the batch size dimension for query, key, value, attention matrix
        global_attention_tokens_indices_padded: torch.Tensor, the indices of the global attention tokens

        window_size: int, the size of local context window
        window_overlap: int, the overlap size between two consecutive windows, same as half of the window_size
        attention_mask: torch.Tensor, mask tensor for masking the padding or causal attention
        attention_dropout: nn.Dropout, dropout layer for the attention scores

    Returns:
        attention_output: torch.Tensor with shape [batch, seq_len, heads*dim_head(=dim_model)]

    References:
        https://github.com/allenai/longformer/blob/master/longformer/longformer.py
        https://github.com/allenai/longformer/blob/master/longformer/sliding_chunks.py
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/longformer/modeling_longformer.py
    """
    # aliasing the tensor dimensions
    batch_size, seq_len, num_heads, dim_head = local_q.size()

    # output shape: batch, seq_len, heads, window_size+1
    local_attention_matrix = sliding_window_attention(local_q, local_k, window_size, window_overlap)

    # values to pad for attention probs
    # if the element of attention_mask is the 0, the element of local_attention_matrix will be the same
    # if the element of attention_mask is the 1, the element of local_attention_matrix will be the -inf
    remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]

    # cast to fp32/fp16 then replace 1's with -inf
    float_mask = remove_from_windowed_attention_mask.type_as(local_q).masked_fill(
        remove_from_windowed_attention_mask, torch.finfo(local_q.dtype).min
    )

    # diagonal mask with zeros everywhere and -inf inplace of padding
    diagonal_mask = sliding_window_attention(
        query=float_mask.new_ones(size=float_mask.size()),
        key=float_mask,
        window_size=window_size,
        window_overlap=window_overlap
    )
    local_attention_matrix += diagonal_mask

    # global attention
    # output shape: [batch, heads, seq_len, nums_global_attention_tokens]
    global_attention_matrix = global_attention(
        global_q,
        global_k,
        row_indices,
        global_attention_tokens_indices_padded
    )

    # concatenate the sliding window attention output & selected tokens global attention output
    # output shape: [batch, seq_len, heads, nums_global_attention_tokens+window_size+1]
    attention_matrix = torch.cat([local_attention_matrix.transpose(2, 1), global_attention_matrix], dim=-1)

    # apply softmax function into row-wise of the attention matrix
    # output shape: [batch, seq_len, heads, nums_global_attention_tokens+window_size+1]
    attention_dist = attention_dropout(
        F.softmax(attention_matrix, dim=-1, dtype=torch.float32)
    )

    # calculate the local attention output and global attention output independently
    # recover the original shape of the attention output: batch, seq_len, heads, dim_head
    attention_output = compute_attn_output_with_global_indices(
        local_v,
        attention_dist,
        row_indices,
        global_attention_matrix.size(-1),
        global_attention_tokens_indices_padded,
        one_sided_attn_window_size=window_overlap,
    ).reshape(-1, seq_len, num_heads * dim_head).contiguous()
    return attention_output


class LongformerMultiHeadAttention(nn.Module):
    def __init__(
        self,
        window_size: int = 512,
        window_overlap: int = 256,
        dim_model: int = 1024,
        num_attention_heads: int = 16,
        dim_head: int = 64,
        attention_dropout_prob: float = 0.1
    ) -> None:
        super(LongformerMultiHeadAttention, self).__init__()
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.dim_model = dim_model
        self.num_attention_heads = num_attention_heads
        self.dim_head = dim_head

        # initialize the linear projection for query, key, value (both local and global)
        self.local_q_proj = nn.Linear(self.dim_model, self.dim_model)
        self.local_k_proj = nn.Linear(self.dim_model, self.dim_model)
        self.local_v_proj = nn.Linear(self.dim_model, self.dim_model)

        self.global_q_proj = nn.Linear(self.dim_model, self.dim_model)
        self.global_k_proj = nn.Linear(self.dim_model, self.dim_model)
        self.global_v_proj = nn.Linear(self.dim_model, self.dim_model)

        self.fc_concat = nn.Linear(self.dim_model, self.dim_model)
        self.attention = longformer_attention
        self.dot_scale = torch.sqrt(torch.tensor(self.dim_head)).to('cuda')
        self.attention_dropout = nn.Dropout(p=attention_dropout_prob)

    def forward(
        self,
        x: Tensor,
        row_indices: Tensor,
        global_attention_tokens_indices_padded: Tensor,
        padding_mask: Tensor,
        attention_mask: Tensor = None
    ) -> Tensor:
        """ x is already passed nn.Layernorm """
        assert x.ndim == 3, f'Expected (batch, seq, hidden) got {x.shape}'

        # linear projection for query, key, value (both local and global)
        # each tensor shape: [batch, seq_len, heads, dim_head]
        local_q = self.local_q_proj(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head)
        local_k = self.local_q_proj(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head)
        local_v = self.local_q_proj(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head)

        global_q = self.global_q_proj(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head)
        global_k = self.global_q_proj(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head)
        global_v = self.global_q_proj(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head)

        attention_output = self.attention(
            local_q=local_q,
            local_k=local_k,
            local_v=local_v,
            global_q=global_q,
            global_k=global_k,
            global_v=global_v,
            row_indices=row_indices,
            global_attention_tokens_indices_padded=global_attention_tokens_indices_padded,
            window_size=self.window_size,
            window_overlap=self.window_overlap,
            attention_mask=padding_mask,
            attention_dropout=self.attention_dropout
        )
        layer_output = self.fc_concat(attention_output)
        return layer_output


class FeedForward(nn.Module):
    """ Class for Feed-Forward Network module in Transformer Encoder Block, this module for BERT
    Same role as Module "BertIntermediate" in official Repo (bert.py)

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


class LongformerEncoderLayer(nn.Module):
    def __init__(
        self,
        window_size: int = 512,
        window_overlap: int = 256,
        dim_model: int = 1024,
        num_attention_heads: int = 16,
        dim_ffn: int = 4096,
        layer_norm_eps: float = 0.02,
        attention_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1
    ) -> None:
        super(LongformerEncoderLayer, self).__init__()
        self.self_attention = LongformerMultiHeadAttention(
            window_size,
            window_overlap,
            dim_model,
            num_attention_heads,
            int(dim_model / num_attention_heads),
            attention_dropout_prob,
        )
        self.layer_norm1 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.hidden_dropout = nn.Dropout(p=hidden_dropout_prob)
        self.ffn = FeedForward(
            dim_model,
            dim_ffn,
            hidden_dropout_prob,
        )

    def forward(
        self,
        x: Tensor,
        row_indices: Tensor,
        global_attention_tokens_indices_padded: Tensor,
        padding_mask: Tensor,
        attention_mask: Tensor = None
    ) -> Tensor:
        ln_x = self.layer_norm1(x)
        residual_x = self.hidden_dropout(
            self.self_attention(ln_x, row_indices, global_attention_tokens_indices_padded, padding_mask, attention_mask)
        ) + x

        ln_x = self.layer_norm2(residual_x)
        fx = self.ffn(ln_x) + residual_x
        return fx


class LongformerEncoder(nn.Module, AbstractModel):
    def __init__(
        self,
        cfg: CFG,
        max_seq: int = 4096,
        window_size: int = 512,
        window_overlap: int = 256,
        num_layers: int = 12,
        dim_model: int = 768,
        num_attention_heads: int = 12,
        dim_ffn: int = 3072,
        layer_norm_eps: float = 0.02,
        attention_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        gradient_checkpointing: bool = False
    ) -> None:
        super(LongformerEncoder, self).__init__()
        self.cfg = cfg
        self.hidden_dropout = nn.Dropout(p=hidden_dropout_prob)  # dropout is not learnable
        self.layer = nn.ModuleList([
            LongformerEncoderLayer(window_size, window_overlap, dim_model, num_attention_heads, dim_ffn, layer_norm_eps, attention_dropout_prob, hidden_dropout_prob) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(dim_model, eps=layer_norm_eps)  # for final-Encoder output
        self.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        inputs: Tensor,
        abs_pos_emb: Tensor,
        row_indices: Tensor,
        global_attention_tokens_indices_padded: Tensor,
        padding_mask: Tensor,
        attention_mask: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: embedding from input sequence
            abs_pos_emb: absolute position embedding
            row_indices: indices of the batch size dimension for query, key, value, attention matrix
            global_attention_tokens_indices_padded: indices of the global attention tokens
            padding_mask: mask for Encoder padded token for speeding up to calculate attention score or MLM
            attention_mask: mask for CLM
        """
        layer_output = []
        x = inputs + abs_pos_emb  # add absolute position embedding with word embedding
        for layer in self.layer:
            if self.gradient_checkpointing and self.cfg.train:
                x = self._gradient_checkpointing_func(
                    layer.__call__,  # same as __forward__ call, torch reference recommend to use __call__ instead of forward
                    x,
                    row_indices,
                    global_attention_tokens_indices_padded,
                    padding_mask,
                    attention_mask
                )
            else:
                x = layer(
                    x,
                    row_indices,
                    global_attention_tokens_indices_padded,
                    padding_mask,
                    attention_mask
                )
            layer_output.append(x)
        last_hidden_state = self.layer_norm(x)  # because of applying pre-layer norm
        hidden_states = torch.stack(layer_output, dim=0).to(x.device)  # shape: [num_layers, BS, SEQ_LEN, DIM_Model]
        return last_hidden_state, hidden_states


class Embedding(nn.Module):
    """ longformer embedding module class
    current embedding type is only absolute positional embedding, same as the original longformer model for mlm task
    ASAP, we will add the RoPE(rotary positional encoding) for this longformer model

    this module initializes the word embedding and absolute positional embedding,
    when this module is called, it returns the word embedding and absolute positional embedding

    from the original longformer paper, the absolute positional embedding is the same as the RoBERTa pretrained weights
    original RoBERTa model has the 512 positional embedding, but longformer has the 4096 positional embedding
    so they copy the first 512 positional embedding from the RoBERTa model and repeat the 512 positional embedding 8 times
    because attention heads strongly learn bias to attending to local context, so repeating the positional embedding is
    quite effective for the long context with local context window. we follow the same strategy for the absolute positional embedding

    Args:
        cfg: configuration object from configuration.py
    """
    def __init__(self, cfg: CFG) -> None:
        super(Embedding, self).__init__()
        self.cfg = cfg
        self.max_seq = cfg.max_seq
        self.word_embedding = nn.Embedding(len(cfg.tokenizer), cfg.dim_model)  # Word Embedding which is not add Absolute Position
        self.abs_pos_emb = nn.Embedding(cfg.max_seq, cfg.dim_model)  # Absolute Position Embedding for EMD Layer
        self.layer_norm1 = nn.LayerNorm(cfg.dim_model, eps=cfg.layer_norm_eps)  # for word embedding
        self.layer_norm2 = nn.LayerNorm(cfg.dim_model, eps=cfg.layer_norm_eps)  # for word embedding
        self.hidden_dropout = nn.Dropout(p=cfg.hidden_dropout_prob)

        # ALBERT Style Factorized Embedding
        if self.cfg.is_mf_embedding:
            self.word_embedding = nn.Embedding(len(cfg.tokenizer), int(cfg.dim_model/6))
            self.projector = nn.Linear(int(cfg.dim_model/6), cfg.dim_model)  # project to original hidden dim

    def forward(self, inputs: Tensor) -> Tuple[nn.Embedding, nn.Embedding]:
        if self.cfg.is_mf_embedding:
            word_embeddings = self.hidden_dropout(
                self.layer_norm1(self.projector(self.word_embedding(inputs)))
            )
        else:
            word_embeddings = self.hidden_dropout(
                self.layer_norm1(self.word_embedding(inputs))
            )
        abs_pos_emb = self.hidden_dropout(
            self.layer_norm2(self.abs_pos_emb(torch.arange(inputs.shape[1], device="cuda").repeat(inputs.shape[0]).view(inputs.shape[0], -1)))
        )
        return word_embeddings, abs_pos_emb


class Longformer(nn.Module, AbstractModel):
    """ main module for Longformer with only encoder module, no decoder module here.
    this module is made for only encoder module, so we doesn't apply the dynamic local context window size
    we just apply the fixed local context window size, and global attention with task-specific tokens
    we doesn't select the dliated sliding window attention because it needs the custom CUDA kernel from the original paper

    This module is made for the MLM task, so we don't apply the CLM task here.

    Also, we apply the same strategy for the absolute positional embedding as the original longformer model,
    which is copying the first 512 positional embedding from the RoBERTa model and repeating the 512 positional embedding 8 times

    Other details are same as the BERT model

    Args:
        cfg: configuration object from configuration.py
        num_layers: int, the number of transformer layers, default 12

    References:

    """
    def __init__(self, cfg: CFG, num_layers: int = 12) -> None:
        super(Longformer, self).__init__()
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.max_seq = cfg.max_seq
        self.window_size = cfg.window_size
        self.window_overlap = cfg.window_overlap
        self.num_layers = num_layers
        self.num_attention_heads = cfg.num_attention_heads
        self.dim_model = cfg.dim_model
        self.dim_ffn = cfg.dim_ffn
        self.layer_norm_eps = cfg.layer_norm_eps
        self.hidden_dropout_prob = cfg.hidden_dropout_prob
        self.attention_dropout_prob = cfg.attention_probs_dropout_prob
        self.gradient_checkpointing = cfg.gradient_checkpoint

        # initialize callable object for making the indices tensor for global attention
        # initialize the Embedding and LongformerEncoder
        self.global_attention_helper = make_global_attention_indices
        self.embeddings = Embedding(cfg)
        self.encoder = LongformerEncoder(
            cfg=cfg,
            max_seq=self.max_seq,
            window_size=self.window_size,
            window_overlap=self.window_overlap,
            num_layers=self.num_layers,
            dim_model=self.dim_model,
            num_attention_heads=self.num_attention_heads,
            dim_ffn=self.dim_ffn,
            layer_norm_eps=self.layer_norm_eps,
            attention_dropout_prob=self.attention_dropout_prob,
            hidden_dropout_prob=self.hidden_dropout_prob,
            gradient_checkpointing=self.gradient_checkpointing
        )

    def forward(self, inputs: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: input sequence, shape (batch_size, sequence)
            padding_mask: padding mask for MLM or padding token
            attention_mask: attention mask for CLM, default None
        """
        assert inputs.ndim == 2, f'Expected (batch, sequence) got {inputs.shape}'

        row_indices, global_attention_tokens_indices_padded = self.global_attention_helper(
            input_ids=inputs,
            global_attention_tokens_id_list=[self.cfg.tokenizer.cls_token_id, self.cfg.tokenizer.sep_token_id]
        )
        word_embeddings, abs_pos_emb = self.embeddings(inputs)
        last_hidden_state, hidden_states = self.encoder(
            inputs=word_embeddings,
            abs_pos_emd=abs_pos_emb,
            row_indices=row_indices,
            global_attention_tokens_indices_padded=global_attention_tokens_indices_padded,
            padding_mask=padding_mask,
            attention_mask=attention_mask
        )
        return last_hidden_state, hidden_states

