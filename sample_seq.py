# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate
from teach.data.sampling.base import FrameSampler
from teach.data.tools.collate import collate_pairs_and_text, collate_datastruct_and_text
from tqdm import tqdm
from teach.data.tools import lengths_to_mask
from teach.render.mesh_viz import visualize_meshes
from teach.render.video import save_video_samples
import sys
from torch.utils.data import DataLoader

def main():
    # dist_util.dev() = 'cpu'
    args = generate_args()
    # args.batch_size = 1
    fixseed(args.seed)
    path = './results/'
    # temppath = path + f"temp.npy"
    # args.tiny = True
    print('Loading dataset...')
    data = load_dataset(args)
    # dataset.dataset = 'babel'
    # total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    # args.tiny = True
    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    align_full_bodies = True
    align_trans = False
    model.sample_mean = False
    model.fact = 1
    slerp_ws = 8
    return_type = "joints"
    transforms = data.dataset.transforms
    transforms.rots2joints.batch_size = 16
    transforms.rots2joints.jointstype = 'mmmns'
    
    with torch.no_grad():
        for batch in tqdm(data):
            motion = forward_seq(args, 
                        model=model,
                        diffusion=diffusion,
                        transforms=transforms,
                        keyid=batch['keyid'],
                        # transforms=transforms,
                        text_0=batch['text_0'],
                        text_1=batch['text_1'],
                        length_0=batch['length_0'],
                        length_1=batch['length_1_with_transition'],
                        align_full_bodies=align_full_bodies,
                        align_only_trans=align_trans,
                        slerp_window_size=slerp_ws,
                        return_type=return_type
                        )


def forward_seq(args, model, diffusion, transforms, keyid, text_0, text_1, length_0, length_1, align_full_bodies=True, align_only_trans=False,
                    slerp_window_size=None, return_type="joints", do_slerp='True'):
    model_kwargs_0 = {}
    model_kwargs_0['y'] = {}
    model_kwargs_0['y']['length'] = length_0 
    model_kwargs_0['y']['text'] = text_0
    model_kwargs_0['y']['mask'] = lengths_to_mask(length_0, dist_util.dev()).unsqueeze(1).unsqueeze(2)
    model_kwargs_0['y']['scale'] = torch.ones(len(text_0), device=dist_util.dev()) * args.guidance_param
    max_length_0 = max(length_0)

    model_kwargs_1 = {}
    model_kwargs_1['y'] = {}
    model_kwargs_1['y']['length'] = [(length - slerp_window_size) for length in length_1] 
    model_kwargs_1['y']['text'] = text_1
    model_kwargs_1['y']['mask'] = lengths_to_mask(model_kwargs_1['y']['length'], dist_util.dev()).unsqueeze(1).unsqueeze(2)
    model_kwargs_1['y']['scale'] = torch.ones(len(text_1), device=dist_util.dev()) * args.guidance_param
    max_length_1 = max(model_kwargs_1['y']['length'])

    sample_fn = diffusion.p_sample_loop_multi

    sample_0, sample_1 = sample_fn(
            model,
            args.hist_frames,
            (len(text_0), model.njoints, model.nfeats, max_length_0),
            (len(text_1), model.njoints, model.nfeats, max_length_1),
            clip_denoised=False,
            model_kwargs_0=model_kwargs_0,
            model_kwargs_1=model_kwargs_1,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
    
    for idx in range(len(text_0)):
        id = keyid[idx]
        feature_0 = sample_0[idx].squeeze().permute(1, 0)
        feature_0 = feature_0[:model_kwargs_0['y']['length'][idx]].cpu()
        feature_1 = sample_1[idx].squeeze().permute(1, 0)
        feature_1 = feature_1[:model_kwargs_1['y']['length'][idx]].cpu()
    
        # sample_0 = sample_0.squeeze().permute(1, 0).cpu()
        # sample_1 = sample_1.squeeze().permute(1, 0).cpu()
        toslerp_inter = torch.tile(0*feature_1[0], (slerp_window_size, 1))
        feature_1 = torch.cat((toslerp_inter, feature_1))
        all_features = torch.cat((feature_0, feature_1), dim=0)

        Datastruct = transforms.Datastruct

        datastruct = Datastruct(features=all_features)

        motion = datastruct.rots
        rots, transl = motion.rots, motion.trans
        pose_rep = "matrix"
        from teach.tools.interpolation import aligining_bodies, slerp_poses, slerp_translation, align_trajectory

        # Rotate bodies etc in place
        lengths = [length_0[idx], length_1[idx]]

        cur_texts = [text_0[idx], text_1[idx]]
        cur_lens = lengths

        end_first_motion = lengths[0] - 1
        for length in lengths[1:]:
            # Compute indices
            begin_second_motion = end_first_motion + 1
            begin_second_motion += slerp_window_size if do_slerp else 0
            # last motion + 1 / to be used with slice
            last_second_motion_ex = end_first_motion + 1 + length

            if align_full_bodies:
                outputs = aligining_bodies(last_pose=rots[end_first_motion],
                                            last_trans=transl[end_first_motion],
                                            poses=rots[begin_second_motion:last_second_motion_ex],
                                            transl=transl[begin_second_motion:last_second_motion_ex],
                                            pose_rep=pose_rep)
                # Alignement
                rots[begin_second_motion:last_second_motion_ex] = outputs[0]
                transl[begin_second_motion:last_second_motion_ex] = outputs[1]
            elif align_only_trans:
                transl[begin_second_motion:last_second_motion_ex] = align_trajectory(transl[end_first_motion],
                                                                                        transl[begin_second_motion:last_second_motion_ex])
            else:
                pass

            # Slerp if needed
            if do_slerp:
                inter_pose = slerp_poses(last_pose=rots[end_first_motion],
                                            new_pose=rots[begin_second_motion],
                                            number_of_frames=slerp_window_size, pose_rep=pose_rep)

                inter_transl = slerp_translation(transl[end_first_motion], transl[begin_second_motion], number_of_frames=slerp_window_size)

                # Fill the gap
                rots[end_first_motion+1:begin_second_motion] = inter_pose
                transl[end_first_motion+1:begin_second_motion] = inter_transl

            # Update end_first_motion
            end_first_motion += length
        from teach.transforms.smpl import RotTransDatastruct
        final_datastruct = Datastruct(rots_=RotTransDatastruct(rots=rots, trans=transl))

        if return_type == "joints":
            motion = final_datastruct.joints
            path = './sample_results/'
            print(f'{id} is processed')
            np.save(path + f"{id}.npy", {'motion': motion.numpy(), 'text': cur_texts, 'lengths': cur_lens} )
            #return final_datastruct.joints


def load_dataset(args):
    if args.dataset != 'babel':
        data = get_dataset_loader(args=args,
                                name=args.dataset,
                                batch_size=args.batch_size,
                                split='test',
                                hml_mode='text_only')
    else:
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
            jointstype='mmm', input_pose_rep='matrix', batch_size=128, gender='male')
        globalvelandy = Globalvelandy(canonicalize=True, pose_rep='rot6d', offset=True,
            path='../teach/deps/transforms/rots2rfeats/globalvelandy/rot6d/babel-amass',
            normalization=True)
        transforms = SMPLTransform(rots2rfeats=globalvelandy, rots2joints=smplh,
            joints2jfeats=rifke)
        sampler= FrameSampler()
        sampler.max_len = 100000
        sampler.min_len = 15
        dataset = BABEL(datapath=datapath, framerate=framerate, dtype=dtype, transforms=transforms, 
            tiny=args.tiny, sampler=sampler, split='val', mode='inference') # mode = inference
        datatype = 'separate_pairs'
        if datatype == 'separate_pairs':
            collate = collate_pairs_and_text
        else:
            collate = collate_datastruct_and_text

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=8, drop_last=False, collate_fn=collate
    )
    # data.fixed_length = n_frames
    return loader


if __name__ == "__main__":
    main()
