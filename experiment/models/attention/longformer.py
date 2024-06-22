import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment.models.abstract_model import AbstractModel
from torch import Tensor
from typing import Tuple, List
from einops.layers.torch import Rearrange
from configuration import CFG


def global_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    row_indices: torch.Tensor,
    global_attention_tokens_indices_padded: torch.Tensor
) -> torch.Tensor:
    """ function for global full attention, applying selected index of token (Task-specific special tokens)
    such as [CLS], [SEP] ...

    Args:

    Returns:

    References:

    """
    selected_q = query[row_indices, global_attention_tokens_indices_padded]
    attention_matrix = torch.matmul(selected_q, key.transpose(-2, -1))
    return attention_matrix


def make_global_attention_indexes(input_ids, global_attention_tokens_id_list: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """ function for building the global attention indexes tensors

    implementations:
        1) create the mask tensor for the special tokens
        2) find the indices of the special tokens
        3) separate the indices for each text
        4) pad the indices to create a tensor of consistent shape
    """
    global_attention_tokens_mask = torch.isin(
        input_ids, torch.tensor(global_attention_tokens_id_list)
    )
    global_attention_tokens_indices = torch.nonzero(global_attention_tokens_mask, as_tuple=False)
    global_attention_tokens_indices_split = torch.split(global_attention_tokens_indices[:, 1], input_ids.size(0))
    global_attention_tokens_indices_padded = torch.nn.utils.rnn.pad_sequence(
        global_attention_tokens_indices_split, batch_first=True, padding_value=-1
    )
    row_indices = torch.arange(
        global_attention_tokens_indices_padded.size(0)
    ).unsqueeze(1).expand_as(global_attention_tokens_indices_padded).to(global_attention_tokens_indices_padded.device)
    return row_indices, global_attention_tokens_indices_padded


def sliding_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    window_size: int,
    window_overlap: int
) -> torch.Tensor:
    """ function for sliding window attention from longformer

    implementation:
        1) make the overlapping chunks of each tensor (query, key)
        2) apply the sliding window attention pattern
        3) recover the original shape of the attention matrix [batch, heads, seq_len, seq_len]
            - for matrix multiplication with value matrix [batch, heads, seq_len, dim_head]

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
    BS, SEQ_LEN, HEADS, DIM_HEAD = query.size()
    chunks_count = torch.div(SEQ_LEN, window_overlap, rounding_mode="trunc") - 1  # number of chunks in current batches

    # integrate the batch size and num_heads dimensions into the seq_len dimension
    query = query.transpose(1, 2).reshape(BS * HEADS, SEQ_LEN, DIM_HEAD)
    key = key.transpose(1, 2).reshape(BS * HEADS, SEQ_LEN, DIM_HEAD)

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
        (BS * HEADS, chunks_count + 1, window_overlap, window_size + 1)
    )
    diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[:, :, :window_overlap, :window_overlap + 1]
    diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[:, -1, window_overlap:, :window_overlap + 1]
    diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[:, :, -(window_overlap + 1):-1, window_overlap + 1:]
    diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[:, 0, :window_overlap-1, 1-window_overlap:]

    # tensor shape: batch, seq_len, heads, window_size+1
    # not yet recover the original shape of the attention matrix
    diagonal_attention_scores = diagonal_attention_scores.view(
        BS,
        HEADS,
        SEQ_LEN,
        window_size + 1
    ).transpose(2, 1)

    _mask_invalid_locations(diagonal_attention_scores, window_overlap)
    return diagonal_attention_scores


def chunk_emb(hidden_states: torch.Tensor, window_size: int) -> torch.Tensor:
    """ convert into overlapping chunks

    Args:
        hidden_states: torch.Tensor with shape [batch*heads, seq_len, dim_head]
        window_size: int, the size of local context window

    workflow:
        1) make the non-overlapping chunks
            - split the seq_len dimension into num_chunks an context_window size

        2) get the tensor-storage meta data for using torch.as_strided() function for sliding window attention
            - change the meta data for the overlapping chunks

        3) apply as_strided function for making overlapping chunks

    References:
        https://github.com/allenai/longformer/blob/master/longformer/longformer.py
        https://github.com/allenai/longformer/blob/master/longformer/sliding_chunks.py
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/longformer/modeling_longformer.py
    """
    # make non-overlapping chunks
    BS_HEADS, SEQ_LEN, DIM_HEAD = hidden_states.size()
    h = hidden_states.view(
        BS_HEADS,
        torch.div(SEQ_LEN, window_size, rounding_mode="trunc"),
        window_size,
        DIM_HEAD
    )

    # use `torch.as_strided()` to make the chunks overlap with an overlap size = window_overlap
    # make overlapping chunks, so number of chunks will be doubled, and stride of the num_chunks dimwill be halved
    chunk_size = list(h.size())  # BS*HEADs, CHUNKS, WINDOW_SIZE, DIM_HEAD
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
    BS_HEADS, CHUNKS, WINDOW_SIZE, DIM_HEAD = h_padded.size()
    h_padded = h_padded.view(
        BS_HEADS,
        CHUNKS,
        DIM_HEAD,
        WINDOW_SIZE
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


def longformer_attention(
    local_q: torch.Tensor,
    local_k: torch.Tensor,
    local_v: torch.Tensor,
    global_q: torch.Tensor,
    global_k: torch.Tensor,
    global_v: torch.Tensor,
    row_indices: torch.Tensor,
    global_attention_tokens_indices_padded: torch.Tensor,
    window_size: int,
    window_overlap: int,
    attention_mask: torch.Tensor = None
):
    """ function for the longformer total attention pattern

    1) sliding window attention
    2) dilated sliding window attention (not supported in general CUDA)
    3) global full attention
    4) apply softmax function into row-wise of the attention matrix

    Args:
        all of the input tensors have the same shape [batch, seq_len, heads, dim_head]

        local_q: projected query tensor from local query projector
        local_k: projected key tensor from local key projector
        local_v: projected value tensor from local value projector

        global_q: projected query tensor from global query projector
        global_k: projected key tensor from global key projector
        global_v: projected value tensor from global value projector
        row_indices: torch.Tensor,
        global_attention_tokens_indices_padded: torch.Tensor

        window_size: int, the size of local context window
        window_overlap: int, the overlap size between two consecutive windows, same as half of the window_size
        attention_mask: torch.Tensor, mask tensor for masking the padding or causal attention

    References:
        https://github.com/allenai/longformer/blob/master/longformer/longformer.py
        https://github.com/allenai/longformer/blob/master/longformer/sliding_chunks.py
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/longformer/modeling_longformer.py
    """
    # shape: batch, seq_len, heads, window_size+1
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
    global_attention_matrix = global_attention(global_q, global_k, row_indices, global_attention_tokens_indices_padded)

    # recover the NxN attention matrix
    batch, num_heads, seq_len, dim_heads = local_q.shape
    attention_matrix = torch.zeros(batch, num_heads, seq_len, seq_len, device=local_q.device)
    attention_matrix[row_indices, global_attention_tokens_indices_padded] = global_attention_matrix

    # need to add the sliding window attention matrix into the global attention matrix
    # we don't need to add the diagonal mask into the global attention matrx
    # because local attention matrix already have the diagonal mask

    return



