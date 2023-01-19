# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate
from teach.data.tools.collate import collate_pairs_and_text, collate_datastruct_and_text, collate_contrastive
from teach.data.sampling.base import FrameSampler

class PretrainDataModule(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 8,
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        # self.dataset = PretrainDataset(root, text_max_len, graph_aug1, graph_aug2)

        from data_loaders.multi_motion.data.dataset import BABEL
        datapath = '../teach/data/babel/babel-smplh-30fps-male'
        framerate = 30
        dtype = 'separate_pairs'
        from teach.transforms.smpl import SMPLTransform
        from teach.transforms.joints2jfeats import Rifke
        from teach.transforms.rots2joints import SMPLH
        from teach.transforms.rots2rfeats import Globalvelandy
        rifke = Rifke(jointstype='mmm', forward_filter=False,
            path='../teach/deps/transforms/joints2jfeats/rifke/babel-amass',
            normalization=True
            )
        smplh = SMPLH(path='../teach/data/smpl_models/smplh',
            jointstype='mmm', input_pose_rep='matrix', batch_size=16, gender='male')
        globalvelandy = Globalvelandy(canonicalize=True, pose_rep='rot6d', offset=True,
            path='../teach/deps/transforms/rots2rfeats/globalvelandy/rot6d/babel-amass',
            # path='../teach/deps/transforms/rots2rfeats/globalvelandy/rot6d/babel-amass/separate_pairs',
            normalization=True)
        transforms = SMPLTransform(rots2rfeats=globalvelandy, rots2joints=smplh,
            joints2jfeats=rifke)
        frame_sampler = FrameSampler()
        frame_sampler.max_len = 256
        tiny = False
        self.dataset = BABEL(datapath=datapath, framerate=framerate, dtype=dtype, transforms=transforms, tiny=tiny, FrameSampler=frame_sampler)
        # datatype = 'separate_pairs'
        self.collate = collate_contrastive

    def setup(self, stage: str = None):
        self.train_dataset = self.dataset

    def train_dataloader(self):
        # loader = DataLoader(
        #     self.train_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     num_workers=self.num_workers,
        #     pin_memory=False,
        #     drop_last=True,
        #     # persistent_workers = True
        # )
        loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True, collate_fn=self.collate
        )
        print('len(train_dataset)', len(self.dataset))
        print('len(train_dataloader)', len(loader))
        return loader