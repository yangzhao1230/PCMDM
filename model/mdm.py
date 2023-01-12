import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
from teach.data.tools import lengths_to_mask

class MDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()

        #self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        self.motion_mask = kargs['motion_mask']
        self.hist_frames = kargs['hist_frames']

        if self.arch == 'past_cond':
            if self.hist_frames > 0:
                # self.hist_frames = 5
                self.seperation_token = nn.Parameter(torch.randn(latent_dim))
                self.skel_embedding = nn.Linear(self.njoints, self.latent_dim)
        if self.arch == 'trans_enc' or self.arch == 'past_cond':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit', 'babel'] else None  # Specific hardcoding for humanml dataset
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

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)

        if self.arch == 'gru':
            x_reshaped = x.reshape(bs, njoints*nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)     #[#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)      #[bs, d, #frames]
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  #[bs, d, 1, #frames]
            x = torch.cat((x_reshaped, emb_gru), axis=1)  #[bs, d+joints*feat, 1, #frames]

        x = self.input_process(x)

        if self.arch == 'trans_enc':
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == 'past_cond':
            mask = lengths_to_mask(y['length'], x.device)
            if self.hist_frames == 0 or y.get('hframes', None) == None:
                token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
                aug_mask = torch.cat((token_mask, mask), 1)
                xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
                xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
                if self.motion_mask:
                    output = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
                else:
                    output = self.seqTransEncoder(xseq)[1:]
            else:
                token_mask = torch.ones((bs, 7), dtype=bool, device=x.device)
                aug_mask = torch.cat((token_mask, mask), 1)
                sep_token = torch.tile(self.seperation_token, (bs,)).reshape(bs, -1).unsqueeze(0)
                hframes = y['hframes'].squeeze(2).permute(2, 0, 1) #TODO find out the diff 
                hframes_emb = self.skel_embedding(hframes)
                # hframes_emb = hframes_emb.permute(1, 0, 2) # [5 b dim]
                xseq = torch.cat((emb, hframes_emb, sep_token, x), axis=0)
                # TODO add attention mask
                xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
                output = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)[7:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
            

        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.emb_trans_dec:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)[1:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)
        elif self.arch == 'gru':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output


    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)
    
    # def forward_seq(self, texts: list[str], lengths: list[int], align_full_bodies=True, align_only_trans=False,
    #                 slerp_window_size=None, return_type="joints"):

    #     assert not (align_full_bodies and align_only_trans)
    #     do_slerp = slerp_window_size is not None
    #     # MUST INCLUDE Z PREV
    #     hframes = None
    #     prev_z = None
    #     all_features = []

    #     for index, (text, length) in enumerate(zip(texts, lengths)):
    #         current_z, _ = self.encode_data([text], hframes=hframes, z_previous=prev_z, return_latent=True)
    #         if do_slerp and index > 0:
    #             length = length - slerp_window_size
    #             assert length > 1

    #         current_features = self.motiondecoder(current_z, lengths=[length])[0]
    #         # create space for slerping
    #         if do_slerp and index > 0:
    #             toslerp_inter = torch.tile(0*current_features[0], (slerp_window_size, 1))
    #             current_features = torch.cat((toslerp_inter, current_features))

    #         all_features.append(current_features)
    #         if self.hist_frames > 0:
    #             hframes = current_features[-self.hist_frames:][None]
    #         if self.previous_latent:
    #             prev_z = current_z
    #     all_features = torch.cat(all_features)
    #     datastruct = self.Datastruct(features=all_features)

    #     motion = datastruct.rots
    #     rots, transl = motion.rots, motion.trans
    #     pose_rep = "matrix"
    #     from teach.tools.interpolation import aligining_bodies, slerp_poses, slerp_translation, align_trajectory

    #     # Rotate bodies etc in place
    #     end_first_motion = lengths[0] - 1
    #     for length in lengths[1:]:
    #         # Compute indices
    #         begin_second_motion = end_first_motion + 1
    #         begin_second_motion += slerp_window_size if do_slerp else 0
    #         # last motion + 1 / to be used with slice
    #         last_second_motion_ex = end_first_motion + 1 + length

    #         if align_full_bodies:
    #             outputs = aligining_bodies(last_pose=rots[end_first_motion],
    #                                        last_trans=transl[end_first_motion],
    #                                        poses=rots[begin_second_motion:last_second_motion_ex],
    #                                        transl=transl[begin_second_motion:last_second_motion_ex],
    #                                        pose_rep=pose_rep)
    #             # Alignement
    #             rots[begin_second_motion:last_second_motion_ex] = outputs[0]
    #             transl[begin_second_motion:last_second_motion_ex] = outputs[1]
    #         elif align_only_trans:
    #             transl[begin_second_motion:last_second_motion_ex] = align_trajectory(transl[end_first_motion],
    #                                                                                  transl[begin_second_motion:last_second_motion_ex])
    #         else:
    #             pass

    #         # Slerp if needed
    #         if do_slerp:
    #             inter_pose = slerp_poses(last_pose=rots[end_first_motion],
    #                                      new_pose=rots[begin_second_motion],
    #                                      number_of_frames=slerp_window_size, pose_rep=pose_rep)

    #             inter_transl = slerp_translation(transl[end_first_motion], transl[begin_second_motion], number_of_frames=slerp_window_size)

    #             # Fill the gap
    #             rots[end_first_motion+1:begin_second_motion] = inter_pose
    #             transl[end_first_motion+1:begin_second_motion] = inter_transl

    #         # Update end_first_motion
    #         end_first_motion += length
    #     from teach.transforms.smpl import RotTransDatastruct
    #     final_datastruct = self.Datastruct(rots_=RotTransDatastruct(rots=rots, trans=transl))

    #     if return_type == "vertices":
    #         return final_datastruct.vertices
    #     elif return_type == "smpl":
    #         return { 'rots': rots, 'transl': transl,
    #                  'vertices': final_datastruct.vertices}
    #     elif return_type == "joints":
    #         return final_datastruct.joints
    #     else:
    #         raise NotImplementedError


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output