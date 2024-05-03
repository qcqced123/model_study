import torch
import torch.nn as nn
import torch.nn.functional as F
import configuration as CFG

from typing import Tuple
from torch import Tensor
from einops.layers.torch import Rearrange
from experiment.models.abstract_model import AbstractModel


def scaled_dot_product_attention(
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
        https://arxiv.org/abs/2010.11929

    """
    BS, NUM_HEADS, SEQ_LEN, DIM_HEADS = q.shape
    attention_matrix = torch.matmul(q, k.permute(0, 2, 3, 1).contiguous()) / dot_scale
    if padding_mask is not None:
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # for broadcasting: shape (BS, 1, 1, SEQ_LEN)
        attention_matrix = attention_matrix.masked_fill(padding_mask == 1, float('-inf'))

    attention_dist = attention_dropout(
        F.softmax(attention_matrix, dim=-1)
    )
    attention_matrix = torch.matmul(attention_dist, v).permute(0, 2, 1, 3).reshape(-1, SEQ_LEN, NUM_HEADS*DIM_HEADS).contiguous()
    return attention_matrix


class MultiHeadAttention(nn.Module):
    """ In this class, we implement workflow of Multi-Head Self-attention for BERT
    This class has same role as Module "BertAttention" in official Repo (bert.py)
    In official repo, they use post-layer norm, but we use pre-layer norm which is more stable & efficient for training

    Args:
        dim_model: dimension of model's latent vector space, default 1024 from official paper
        num_attention_heads: number of heads in MHSA, default 16 from official paper for Transformer
        dim_head: dimension of each attention head, default 64 from official paper (1024 / 16)
        attention_dropout_prob: dropout rate, default 0.1

    Math:
        A = softmax(attention Matrix/sqrt(3*D_h)), SA(z) = Av

    Reference:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/pdf/1810.04805.pdf
        https://arxiv.org/abs/2010.11929

    """
    def __init__(self, dim_model: int = 1024, num_attention_heads: int = 16, dim_head: int = 64, attention_dropout_prob: float = 0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        self.dim_model = dim_model
        self.num_attention_heads = num_attention_heads
        self.dim_head = dim_head
        self.fc_q = nn.Linear(self.dim_model, self.dim_model)
        self.fc_k = nn.Linear(self.dim_model, self.dim_model)
        self.fc_v = nn.Linear(self.dim_model, self.dim_model)
        self.fc_concat = nn.Linear(self.dim_model, self.dim_model)
        self.attention = scaled_dot_product_attention
        self.dot_scale = torch.sqrt(torch.tensor(self.dim_head)).to('cuda')
        self.attention_dropout = nn.Dropout(p=attention_dropout_prob)

    def forward(self, x: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tensor:
        """ x is already passed nn.Layernorm """
        assert x.ndim == 3, f'Expected (batch, seq, hidden) got {x.shape}'

        # size: bs, seq, nums head, dim head, linear projection
        q = self.fc_q(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        k = self.fc_k(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head)
        v = self.fc_v(x).reshape(-1, x.shape[1], self.num_attention_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        attention_matrix = self.attention(
            q,
            k,
            v,
            self.dot_scale,
            self.attention_dropout,
            padding_mask,
            attention_mask
        )
        attention_output = self.fc_concat(attention_matrix)
        return attention_output


class FeedForward(nn.Module):
    """ Class for Feed-Forward Network module in Transformer Encoder Block, this module for ViT
    Same Role as "MLP" in official paper and Repo

    Args:
        dim_model: dimension of model's latent vector space, default 1024
        dim_ffn: dimension of FFN's hidden layer, default 4096 from official paper
        hidden_dropout_prob: dropout rate, default 0.1

    Math:
        FeedForward(x) = FeedForward(LN(x))+x

    References:
        https://arxiv.org/abs/2010.11929

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


class VisionEncoderLayer(nn.Module):
    """ Abstract Module for single ViT encoder layer, We apply Pre-Layernorm and activation for gradient flow stability

    """
    def __init__(
        self,
        dim_model: int = 1024,
        num_attention_heads: int = 16,
        dim_mlp: int = 4096,
        layer_norm_eps: float = 0.02,
        attention_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1
    ) -> None:
        super(VisionEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            dim_model,
            num_attention_heads,
            int(dim_model / num_attention_heads),
            attention_dropout_prob,
        )
        self.self_attention = MultiHeadAttention(
            dim_model,
            num_attention_heads,
            int(dim_model / num_attention_heads),
            attention_dropout_prob,
        )
        self.layer_norm1 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.hidden_dropout = nn.Dropout(p=hidden_dropout_prob)
        self.mlp = FeedForward(
            dim_model,
            dim_mlp,
            hidden_dropout_prob,
        )

    def forward(self, x: Tensor) -> Tensor:
        ln_x = self.layer_norm1(x)
        residual_x = self.hidden_dropout(self.self_attention(ln_x)) + x

        ln_x = self.layer_norm2(residual_x)
        fx = self.mlp(ln_x) + residual_x  # from official paper & code by Google Research
        return fx


class VisionEncoder(nn.Module, AbstractModel):
    """ In this class, encode input sequence(Image) and then we stack N VisionEncoderLayer

    This model is implemented by cls pooling method for classification

    First, we define "positional embedding" and then add to input embedding for making patch embedding
    Second, forward patch embedding to N EncoderLayer and then get output embedding

    Args:
        num_patches: number of patches in input image => (image_size / patch_size)**2
        num_layers: number of EncoderLayer, default 24 for large model
    """

    def __init__(
        self,
        cfg: CFG,
        num_patches: int,
        patch_size: int,
        num_layers: int = 24,
        num_attention_heads: int = 16,
        dim_model: int = 1024,
        dim_mlp: int = 4096,
        layer_norm_eps: float = 0.02,
        hidden_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        gradient_checkpointing: bool = False
    ) -> None:
        super(VisionEncoder, self).__init__()
        self.cfg = cfg
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_attention_heads
        self.dim_model = dim_model
        self.dim_mlp = dim_mlp
        self.hidden_dropout = nn.Dropout(p=hidden_dropout_prob)  # dropout is not learnable
        self.layer = nn.ModuleList(
            [VisionEncoderLayer(dim_model, num_attention_heads, dim_mlp, layer_norm_eps, attention_dropout_prob, hidden_dropout_prob) for _ in range(self.num_layers)]
        )
        self.layer_norm = nn.LayerNorm(dim_model, eps=layer_norm_eps)  # for final-Encoder output
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, inputs: Tensor, abs_pos_emb: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: embedding from input sequence
            abs_pos_emb: absolute position embedding
        """
        layer_output = []
        x = inputs + abs_pos_emb  # add absolute position embedding with word embedding
        for layer in self.layer:
            if self.gradient_checkpointing and self.cfg.train:
                x = self._gradient_checkpointing_func(
                    layer.__call__,
                    x,
                )
            else:
                x = layer(
                    x,
                )
            layer_output.append(x)
        last_hidden_state = self.layer_norm(x)  # because of applying pre-layer norm
        hidden_states = torch.stack(layer_output, dim=0).to(x.device)  # shape: [num_layers, BS, SEQ_LEN, DIM_Model]
        return last_hidden_state, hidden_states


class Embedding(nn.Module):
    """ ViT Embedding Module class

    This Module set & initialize 2 Embedding Layers:
        1) Flatten Patch Embedding
        2) Absolute Positional Embedding

    Args:
        cfg: configuration.py
    """
    def __init__(self, cfg: CFG) -> None:
        super(Embedding, self).__init__()
        self.cfg = cfg
        self.max_seq = cfg.num_patches + 1
        self.image_slicer = nn.Conv2d(cfg.channels, cfg.dim_model, cfg.patch_size, cfg.patch_size)
        self.patch_emb = nn.Linear(cfg.patch_size**2 * cfg.channels, cfg.dim_model)
        self.abs_pos_emb = nn.Linear(self.max_seq, cfg.dim_model)  # add 1 for cls token
        self.layer_norm1 = nn.LayerNorm(cfg.dim_model, eps=cfg.layer_norm_eps)  # for word embedding
        self.layer_norm2 = nn.LayerNorm(cfg.dim_model, eps=cfg.layer_norm_eps)  # for word embedding
        self.hidden_dropout = nn.Dropout(p=cfg.hidden_dropout_prob)

    def forward(self, inputs: Tensor) -> Tuple[nn.Linear, nn.Linear]:
        assert inputs.ndim != 4, f"Input shape should be [BS, CHANNEL, IMAGE_SIZE, IMAGE_SIZE], but got {inputs.shape}"

        patch_x = self.image_slicer(inputs)
        patch_emb = self.hidden_dropout(
            self.layer_norm1(self.patch_emb(patch_x))
        )
        abs_pos_emb = self.hidden_dropout(
            self.layer_norm2(self.abs_pos_emb(
                torch.arange(self.max_seq, device="cuda").repeat(inputs.shape[0]).view(inputs.shape[0], -1)))
        )
        return patch_emb, abs_pos_emb


class VisionTransformer(nn.Module):
    """ Main class for ViT of cls pooling, Pytorch implementation

    We apply nn.Conv2d for making patches from original image, this method is much simpler than using nn.Linear logic
    nn.Linear() logic must split and view original tensor in other ways. that method needs to too much hard work

    input must be [BS, CHANNEL, IMAGE_SIZE, IMAGE_SIZE]
    In NLP, input_sequence is always smaller than vocab size

    But in vision, input_sequence is always same as image size, not concept of vocab in vision
    So, ViT use nn.Linear instead of nn.Embedding for input_embedding

    Args:
        num_classes: number of classes for classification task
        image_size: size of input image, default 512
        patch_size: size of patch, default 16 from official paper for ViT-Large
        extractor: option for feature extractor, default 'base' which is crop & just flatten with Linear Projection
                   if you want to use convolution for feature extractor, set extractor='cnn' named hybrid ver in paper
        classifier: option for pooling method, default token meaning that do cls pooling
                    if you want to use mean pooling, set classifier='mean'
        mode: option for train type, default fine-tune, if you want pretrain, set mode='pretrain'
              In official paper & code by Google Research, they use different classifier head for pretrain, fine-tune

    Math:
        image2sequence: [batch, channel, image_size, image_size] -> [batch, patch, patch_size^2*channel]
        input_embedding: R^(P^2 ·C)×D

    Reference:
        https://arxiv.org/abs/2010.11929
        https://arxiv.org/abs/1706.03762
        https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py#L184
    """
    def __init__(self, cfg: CFG, num_layers: int = 12) -> None:
        super(VisionTransformer, self).__init__()
        self.cfg = cfg
        self.num_patches = cfg.num_patches
        self.patch_size = cfg.patch_size
        self.num_layers = num_layers
        self.num_attention_heads = cfg.num_attention_heads
        self.dim_model = cfg.dim_model
        self.dim_mlp = cfg.dim_mlp
        self.layer_norm_eps = cfg.layer_norm_eps
        self.hidden_dropout_prob = cfg.hidden_dropout_prob
        self.attention_dropout_prob = cfg.attention_probs_dropout_prob
        self.gradient_checkpointing = cfg.gradient_checkpoint

        # Input Embedding Layer
        self.embeddings = Embedding(cfg)

        # Encoder Multi-Head Self-attention
        self.encoder = VisionEncoder(
            self.cfg,
            self.num_patches,
            self.patch_size,
            self.num_layers,
            self.num_attention_heads,
            self.dim_model,
            self.dim_mlp,
            self.layer_norm_eps,
            self.hidden_dropout_prob,
            self.attention_dropout_prob,
            self.gradient_checkpointing,
        )
        self.pretrain_classifier = nn.Sequential(
            nn.Linear(self.dim_model, self.dim_model),
            nn.Tanh(),
        )

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """ For cls pooling """
        assert inputs.ndim != 4, f"Input shape should be [BS, CHANNEL, IMAGE_SIZE, IMAGE_SIZE], but got {inputs.shape}"
        patch_emb, abs_pos_emb = self.embeddings(inputs)

        last_hidden_state, hidden_states = self.encoder(
            patch_emb,
            abs_pos_emb,
        )
        return last_hidden_state, hidden_states

