import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from typing import List
import torch
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import math
from transformers import CLIPTokenizer, CLIPTextModel

class TextEncoder(nn.Module):
    def __init__(
        self,
        # clip_version="ViT-B/32"
    ):
        super().__init__()
        # self.clip_version = clip_version
        self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, raw_text):

        device = next(self.parameters()).device
        text = self.tokenizer(raw_text, padding='max_length', return_tensors="pt", max_length=22, truncation=True)
        input_ids = text["input_ids"]
        attention_mask = text["attention_mask"]
        output = self.clip_model(input_ids.to(device), attention_mask.to(device))['pooler_output'] 
        return output