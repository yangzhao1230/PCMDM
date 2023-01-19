import torch
import torch.nn as nn
from model.motion_encoder import MotionEncoder
from model.text_encoder import TextEncoder
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim

class MotionClip(pl.LightningModule):
    def __init__(
            self,
            temperature,
            motion_hidden_dim,
            text_hidden_dim,
            projection_dim,
            lr,
            weight_decay,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.temperature = temperature
        
        self.motion_hidden_dim = motion_hidden_dim
        self.text_hidden_dim = text_hidden_dim

        self.projection_dim = projection_dim

        self.lr = lr
        self.weight_decay = weight_decay

        self.motion_encoder = MotionEncoder()
        self.text_encoder = TextEncoder()

        self.motion_proj_head = nn.Sequential(
          nn.Linear(self.motion_hidden_dim, self.motion_hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.motion_hidden_dim, self.projection_dim)
        )
        self.text_proj_head = nn.Sequential(
          nn.Linear(self.text_hidden_dim, self.text_hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.text_hidden_dim, self.projection_dim)
        )

    def forward(self, features_motion, features_text):
        batch_size = features_motion.size(0)

        # normalized features
        features_motion = F.normalize(features_motion, dim=-1)
        features_text = F.normalize(features_text, dim=-1)

        # cosine similarity as logits
        logits_per_motion = features_motion @ features_text.t() / self.temperature
        logits_per_text = logits_per_motion.t()

        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)  # 大小为B
        loss_motion = F.cross_entropy(logits_per_motion, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_motion + loss_text) / 2

        return logits_per_motion, logits_per_text, loss

    def configure_optimizers(self):
        # High lr because of small dataset and small model
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        
        # batch = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in batch.items()}
        motion_feats = batch["motion_feats"]
        length = batch["length"]
        text = batch["text"]

        motion_rep = self.motion_encoder(motion_feats, length)
        motion_rep = self.motion_proj_head(motion_rep)

        text_rep = self.text_encoder(text)
        text_rep = self.text_proj_head(text_rep)

        _, _, loss = self.forward(motion_rep, text_rep)

        self.log("train_loss", loss)

        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MotionClip")
        # train mode
        parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')

        parser.add_argument('--motion_hidden_dim', type=int, default=256, help='')

        parser.add_argument('--text_hidden_dim', type=int, default=512, help='')

        parser.add_argument('--projection_dim', type=int, default=256)

        # optimization
        parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay')

        return parent_parser

