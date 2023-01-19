import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from typing import List
import torch
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import math
from teach.data.tools import lengths_to_mask

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class MotionEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int = 8,
        n_heads: int = 4,
        input_dim: int = 135,
        embed_dim: int = 256,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_motion_length: int = 512,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_motion_length = max_motion_length

        self.input_projection = nn.Linear(input_dim, embed_dim)
        # transformer blocks
        encoder_layers = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # self.motion_length_emb = nn.Embedding(max_motion_length, embed_dim)
        self.motion_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.CLS_TOKEN = nn.Parameter(torch.randn(1, 1, embed_dim))
        # self.last_projection = nn.Linear(embed_dim, lm_hidden_size)

    def forward(self, motion: torch.Tensor, motion_length: List[int]):

        # motion = motion.permute(0, 2, 1)  # (N,C,L) -> (N,T,C)
        device = motion.device
        mask = lengths_to_mask(motion_length, device)
        B, T, C = motion.shape

        cls_token = self.CLS_TOKEN.repeat((B, 1, 1))

        # motion length embedding
        # ml_token = self.motion_length_emb(motion_length)[:, None, :]

        motion_input_proj = self.input_projection(motion)  # (N,L,E)
        input_tokens = torch.cat(
            [cls_token, motion_input_proj], dim=1
        )  # [B, T + 2, E]

        pos_enc = timestep_embedding(
            torch.arange(T + 1, device=device), self.embed_dim
        ).repeat((B, 1, 1))
        input_tokens_pe = input_tokens + pos_enc

        mask_cls_len = torch.ones(
            (B, 1), dtype=bool, device=device
        )  # extend mask for CLS token and length token
        mask_ext = torch.cat([mask_cls_len, mask], dim=1)

        out = self.motion_encoder(
            src=input_tokens_pe, src_key_padding_mask=~mask_ext
        )  # mask_ext: (N,T+1)
        # out = self.last_projection(encoder_out)

        output_dict = {
            "pooler_output": out[:, 0, :],
            "last_hidden_state": out,
        }
        return output_dict["pooler_output"]