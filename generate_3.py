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

def generate_mask(shape, prob):
    mask = torch.bernoulli(torch.ones(shape) * prob)
    mask = torch.tensor(mask, dtype=torch.bool)
    return mask
    #print(mask)    

# generate_mask((1,10,1,10), 0.1)

def main():
    # dist_util.dev() = 'cpu'
    args = generate_args()
    args.batch_size = 1
    fixseed(args.seed)
    path = './results/'
    # temppath = path + f"temp.npy"
    args.tiny = True
    print('Loading dataset...')
    dataset = load_dataset(args)
    dataset.dataset = 'babel'
    # total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, dataset)

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

    transforms = dataset.transforms
    # transforms.rots2joints.jointstype = 'mmmns'

    texts = ['throw ball', 'walk like a drunk', 'fall down']

    texts_list = (['walk in circle', 'sit down'],
                ['throw', 'catch'],
                ['climb down ladder', 'steps left'],
                ['sit cross legs', 'stand'],
                ['walk', 'sit down'],
                ['stand', 'walk like a drink person'],
                ['step forward with right foot', 'kick with left foot'],
                ['dance ballet', 'walk'],
                ['pick something with right hand', 'place it'],
                ['walk in circle', 'sit down'],
                ['walk in circle', 'sit down'],
                ['wave the right hand', 'raise the left hand'],
                ['walk in circle', 'sit down'],
                ['hold a golf club while look at the ground', 'swing golf club'])
    
    # for text in texts_list:

    file_name = texts[0] + '_' + texts[1] + '_ ' + texts[2]
    lengths = [45, 45, 45]
    slerp_ws = 0
    return_type="smpl"
    motion = forward_seq(args, 
                            model=model,
                            diffusion=diffusion,
                            transforms=transforms,
                            texts=texts, 
                            lengths=lengths,
                            align_full_bodies=align_full_bodies,
                            align_only_trans=align_trans,
                            slerp_window_size=slerp_ws,
                            return_type=return_type)
    # np.save(f'video/results_mp4.npy',
    #         {'vertices': motion['vertices'].numpy(),
    #             'rots': motion['rots'].numpy(),
    #             'transl': motion['transl'].numpy(),
    #             'text': texts,
    #             'lengths': lengths} 
    #         ) 
    motion = motion['vertices'].numpy()
    vid_ = visualize_meshes(motion)
    save_video_samples(vid_, f'save/video/{file_name}.mp4', texts, fps=30)


def forward_seq(args, model, diffusion, transforms, texts, lengths, align_full_bodies=True, align_only_trans=False,
                    slerp_window_size=None, return_type="joints", do_slerp='True'):
    # slerp_window_size = 0
    model_kwargs_0 = {}
    model_kwargs_0['y'] = {}
    model_kwargs_0['y']['length'] = [lengths[0]] 
    model_kwargs_0['y']['text'] = [texts[0]]
    model_kwargs_0['y']['mask'] = lengths_to_mask([lengths[0]], dist_util.dev()).unsqueeze(1).unsqueeze(2)
    model_kwargs_0['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

    model_kwargs_1 = {}
    model_kwargs_1['y'] = {}
    model_kwargs_1['y']['length'] = [lengths[1] + args.inpainting_frames] 
    model_kwargs_1['y']['text'] = [texts[1]]
    model_kwargs_1['y']['mask'] = lengths_to_mask([lengths[1] + args.inpainting_frames], dist_util.dev()).unsqueeze(1).unsqueeze(2)
    model_kwargs_1['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

    sample_fn = diffusion.p_sample_loop_inpainting


    sample_0, sample_1 = sample_fn(
            model,
            args.inpainting_frames,
            (args.batch_size, model.njoints, model.nfeats, lengths[0]),
            (args.batch_size, model.njoints, model.nfeats, lengths[1] + args.inpainting_frames),
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
    # print(sample_1.shape)
    if args.inpainting_frames > 0:
        sample_1 = sample_1[:,:,:,args.inpainting_frames:] # [bs 135 1 len] 


    model_kwargs_2 = {}
    model_kwargs_2['y'] = {}
    model_kwargs_2['y']['length'] = [lengths[2] + args.inpainting_frames] 
    model_kwargs_2['y']['text'] = [texts[2]]
    model_kwargs_2['y']['mask'] = lengths_to_mask([lengths[2] + args.inpainting_frames], dist_util.dev()).unsqueeze(1).unsqueeze(2)
    model_kwargs_2['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

    if args.inpainting_frames > 0:
            model_kwargs_2['y']['hist_motion'] = sample_1[:,:,:,-args.inpainting_frames:]

    sample_fn_nxt = diffusion.p_sample_loop

    sample_2 = sample_fn_nxt(
        model,
        (args.batch_size, model.njoints, model.nfeats, lengths[2] + args.inpainting_frames),
        clip_denoised=False,
        model_kwargs=model_kwargs_2,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )
    if args.inpainting_frames > 0:
        sample_2 = sample_2[:,:,:,args.inpainting_frames:] # [bs 135 1 len]

    sample_0 = sample_0.squeeze().permute(1, 0).cpu()
    sample_1 = sample_1.squeeze().permute(1, 0).cpu()
    sample_2 = sample_2.squeeze().permute(1, 0).cpu()
    
    all_features = torch.cat((sample_0, sample_1, sample_2), dim=0)

    Datastruct = transforms.Datastruct

    datastruct = Datastruct(features=all_features)

    motion = datastruct.rots
    rots, transl = motion.rots, motion.trans
    pose_rep = "matrix"
    from teach.tools.interpolation import aligining_bodies, slerp_poses, slerp_translation, align_trajectory

    from teach.transforms.smpl import RotTransDatastruct
    final_datastruct = Datastruct(rots_=RotTransDatastruct(rots=rots, trans=transl))

    if return_type == "vertices":
        return final_datastruct.vertices
    elif return_type == "smpl":
        return { 'rots': rots, 'transl': transl,
                    'vertices': final_datastruct.vertices}
    elif return_type == "joints":
        return final_datastruct.joints
    else:
        raise NotImplementedError

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
            jointstype='mmm', input_pose_rep='matrix', batch_size=16, gender='male')
        globalvelandy = Globalvelandy(canonicalize=True, pose_rep='rot6d', offset=True,
            path='../teach/deps/transforms/rots2rfeats/globalvelandy/rot6d/babel-amass',
            normalization=True)
        transforms = SMPLTransform(rots2rfeats=globalvelandy, rots2joints=smplh,
            joints2jfeats=rifke)
        sampler= FrameSampler()
        sampler.max_len = 100000
        sampler.min_len = 15
        dataset = BABEL(datapath=datapath, framerate=framerate, dtype=dtype, transforms=transforms, 
            tiny=args.tiny, sampler=sampler, split='val', mode='inference')
        datatype = 'separate_pairs'
        if datatype == 'separate_pairs':
            collate = collate_pairs_and_text
        else:
            collate = collate_datastruct_and_text
    # data.fixed_length = n_frames
    return dataset


if __name__ == "__main__":
    main()
