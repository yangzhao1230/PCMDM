import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from typing import List
import torch
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import math
from transformers import CLIPTokenizer, CLIPTextModel
import clip

class TextEncoder(nn.Module):
    def __init__(
        self,
        clip_version="ViT-B/32"
    ):
        super().__init__()
        self.clip_version = clip_version
        self.clip_model, _ = clip.load(clip_version, device='cpu', jit=False)  # Must set jit=False for training


    def forward(self, raw_text):

        device = next(self.parameters()).device
        max_text_len = 20
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()