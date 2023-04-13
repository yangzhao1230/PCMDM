import torch
from data_loaders.humanml.networks.modules import *
from data_loaders.humanml.networks.trainers import CompTrainerV6
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm
from utils import dist_util
from teach.data.tools import lengths_to_mask 
import numpy as np

def build_models(opt):
    if opt.text_enc_mod == 'bigru':
        text_encoder = TextEncoderBiGRU(word_size=opt.dim_word,
                                        pos_size=opt.dim_pos_ohot,
                                        hidden_size=opt.dim_text_hidden,
                                        device=opt.device)
        text_size = opt.dim_text_hidden * 2
    else:
        raise Exception('Text Encoder Mode not Recognized!!!')

    seq_prior = TextDecoder(text_size=text_size,
                            input_size=opt.dim_att_vec + opt.dim_movement_latent,
                            output_size=opt.dim_z,
                            hidden_size=opt.dim_pri_hidden,
                            n_layers=opt.n_layers_pri)


    seq_decoder = TextVAEDecoder(text_size=text_size,
                                 input_size=opt.dim_att_vec + opt.dim_z + opt.dim_movement_latent,
                                 output_size=opt.dim_movement_latent,
                                 hidden_size=opt.dim_dec_hidden,
                                 n_layers=opt.n_layers_dec)

    att_layer = AttLayer(query_dim=opt.dim_pos_hidden,
                         key_dim=text_size,
                         value_dim=opt.dim_att_vec)

    movement_enc = MovementConvEncoder(opt.dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, opt.dim_pose)

    len_estimator = MotionLenEstimatorBiGRU(opt.dim_word, opt.dim_pos_ohot, 512, opt.num_classes)

    # latent_dis = LatentDis(input_size=opt.dim_z * 2)
    checkpoints = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_est_bigru', 'model', 'latest.tar'), map_location=opt.device)
    len_estimator.load_state_dict(checkpoints['estimator'])
    len_estimator.to(opt.device)
    len_estimator.eval()

    # return text_encoder, text_decoder, att_layer, vae_pri, vae_dec, vae_pos, motion_dis, movement_dis, latent_dis
    return text_encoder, seq_prior, seq_decoder, att_layer, movement_enc, movement_dec, len_estimator

class CompV6GeneratedDataset(Dataset):

    def __init__(self, opt, dataset, w_vectorizer, mm_num_samples, mm_num_repeats):
        assert mm_num_samples < len(dataset)
        print(opt.model_dir)

        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
        text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec, len_estimator = build_models(opt)
        trainer = CompTrainerV6(opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, mov_enc=mov_enc)
        epoch, it, sub_ep, schedule_len = trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))
        generated_motion = []
        mm_generated_motions = []
        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)
        min_mov_length = 10 if opt.dataset_name == 't2m' else 6
        # print(mm_idxs)

        print('Loading model: Epoch %03d Schedule_len %03d' % (epoch, schedule_len))
        trainer.eval_mode()
        trainer.to(opt.device)
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
                tokens = tokens[0].split('_')
                word_emb = word_emb.detach().to(opt.device).float()
                pos_ohot = pos_ohot.detach().to(opt.device).float()

                pred_dis = len_estimator(word_emb, pos_ohot, cap_lens)
                pred_dis = nn.Softmax(-1)(pred_dis).squeeze()

                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False

                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):
                    mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)

                    m_lens = mov_length * opt.unit_length
                    pred_motions, _, _ = trainer.generate(word_emb, pos_ohot, cap_lens, m_lens,
                                                          m_lens[0]//opt.unit_length, opt.dim_pose)
                    if t == 0:
                        # print(m_lens)
                        # print(text_data)
                        sub_dict = {'motion': pred_motions[0].cpu().numpy(),
                                    'length': m_lens[0].item(),
                                    'cap_len': cap_lens[0].item(),
                                    'caption': caption[0],
                                    'tokens': tokens}
                        generated_motion.append(sub_dict)

                    if is_mm:
                        mm_motions.append({
                            'motion': pred_motions[0].cpu().numpy(),
                            'length': m_lens[0].item()
                        })
                if is_mm:
                    mm_generated_motions.append({'caption': caption[0],
                                                 'tokens': tokens,
                                                 'cap_len': cap_lens[0].item(),
                                                 'mm_motions': mm_motions})

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.opt = opt
        self.w_vectorizer = w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if m_length < self.opt.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.opt.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

class CompMDMGeneratedDataset(Dataset):

    def __init__(self, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1.):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        self.max_motion_length = max_motion_length
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()


        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                # add CFG scale to batch
                if scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
                                                            device=dist_util.dev()) * scale

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):

                    sample = sample_fn(
                        model,
                        motion.shape,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=None,
                        progress=False,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )

                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                    'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        } for bs_i in range(dataloader.batch_size)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']

        if self.dataset.mode == 'eval':
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


class CompCCDGeneratedDataset(Dataset):

    def __init__(self, args, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, num_samples_limit, scale=1.):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        # self.max_motion_length = max_motion_length
        sample_fn = (
            diffusion.p_sample_loop_inpainting if not use_ddim else diffusion.ddim_sample_loop
        )
        if args.refine:
            sample_fn_refine = diffusion.p_sample_loop

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()


        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader)):

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break
                bs = len(batch['length_0'])

                model_kwargs_0 = {}
                model_kwargs_0['y'] = {}
                model_kwargs_0['y']['length'] = batch['length_0']
                model_kwargs_0['y']['text'] = batch['text_0']
                model_kwargs_0['y']['mask'] = lengths_to_mask(model_kwargs_0['y']['length'], 
                                    dist_util.dev()).unsqueeze(1).unsqueeze(2)

                model_kwargs_1 = {}
                model_kwargs_1['y'] = {}
                model_kwargs_1['y']['length'] = [args.inpainting_frames + len 
                                                for len in batch['length_1_with_transition']]
                model_kwargs_1['y']['text'] = batch['text_1']
                model_kwargs_1['y']['mask'] = lengths_to_mask(model_kwargs_1['y']['length'], 
                                    dist_util.dev()).unsqueeze(1).unsqueeze(2)
                # add CFG scale to batch
                if scale != 1.:
                    model_kwargs_0['y']['scale'] = torch.ones(len(model_kwargs_0['y']['length']),
                                                            device=dist_util.dev()) * scale
                    model_kwargs_1['y']['scale'] = torch.ones(len(model_kwargs_1['y']['length']),
                                                            device=dist_util.dev()) * scale

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []

                for t in range(repeat_times):
                    sample_0, sample_1 = sample_fn(
                        model,
                        args.inpainting_frames,
                        (bs, 135, 1, model_kwargs_0['y']['mask'].shape[-1]),
                        (bs, 135, 1, model_kwargs_1['y']['mask'].shape[-1]),
                        clip_denoised=clip_denoised,
                        model_kwargs_0=model_kwargs_0,
                        model_kwargs_1=model_kwargs_1,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=None,
                        progress=True,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )
                    if args.inpainting_frames > 0:
                        sample_1 = sample_1[:,:,:,args.inpainting_frames:] # B 135 1 L
                    if args.refine:
                        model_kwargs_0_refine = {}
                        model_kwargs_0_refine['y'] = {}
                        model_kwargs_0_refine['y']['scale'] = model_kwargs_0['y']['scale']
                        model_kwargs_0_refine['y']['text'] = model_kwargs_0['y']['text']
                        model_kwargs_0_refine['y']['next_motion'] = sample_1[:,:,:,:args.inpainting_frames]
                        model_kwargs_0_refine['y']['length'] = [len + args.inpainting_frames
                                                        for len in model_kwargs_0['y']['length']]
                        model_kwargs_0_refine['y']['mask'] = lengths_to_mask(model_kwargs_0_refine['y']['length'], dist_util.dev()).unsqueeze(1).unsqueeze(2)
                        sample_0_refine = sample_fn_refine( # bs 135 1 len+inpainting 
                                                    model,
                                                    (bs, 135, 1, model_kwargs_0_refine['y']['mask'].shape[-1]),
                                                    clip_denoised=False,
                                                    model_kwargs=model_kwargs_0_refine,
                                                    skip_timesteps=0, 
                                                    init_image=None,
                                                    progress=True,
                                                    dump_steps=None,
                                                    noise=None,
                                                    const_noise=False)
                        sample_0_refine = sample_0_refine[:,:,:,:-args.inpainting_frames]
                        sample_0 = sample_0  + args.refine_scale * (sample_0_refine - sample_0)
                        # model_kwargs_0['y']['length'] = [len - args.inpainting_frames
                        #                                 for len in model_kwargs_0['y']['length']]

                    sample_0 = sample_0.squeeze().permute(0, 2, 1).cpu().numpy() # B L D
                    sample_1 = sample_1.squeeze().permute(0, 2, 1).cpu().numpy()
                    length_0 = batch['length_0']
                    length_1 = batch['length_1_with_transition']
                    length = [length_0[idx] + length_1[idx] for idx in range(bs)]
                    def collate_tensor_with_padding(batch):
                        batch = [torch.tensor(x) for x in batch]
                        dims = batch[0].dim()
                        max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
                        size = (len(batch),) + tuple(max_size)
                        canvas = batch[0].new_zeros(size=size)
                        for i, b in enumerate(batch):
                            sub_tensor = canvas[i]
                            for d in range(dims):
                                sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
                            sub_tensor.add_(b)
                        canvas = canvas.detach().numpy()
                        # canvas = [x.detach().numpy() for x in canvas]
                        return canvas

                    def merge(motion_0, length_0, motion_1, length_1): # B L D
                        bs = motion_0.shape[0]
                        ret = []
                        for idx in range(bs):
                             ret.append(np.concatenate((motion_0[idx,:length_0[idx]], motion_1[idx,:length_1[idx]]), axis=0))
                        
                        return collate_tensor_with_padding(ret)

                    sample = merge(sample_0, length_0, sample_1, length_1)
                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i],
                                    'length': length[bs_i],
                                    'text': model_kwargs_0['y']['text'][bs_i] + ', ' +  model_kwargs_1['y']['text'][bs_i]
                                    # 'caption': model_kwargs['y']['text'][bs_i],
                                    # 'tokens': tokens[bs_i],
                                    # 'cap_len': len(tokens[bs_i]),
                                    } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts
                    # if t == 0:
                    #     sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                    #                 'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                    #                 'caption': model_kwargs['y']['text'][bs_i],
                    #                 # 'tokens': tokens[bs_i],
                    #                 # 'cap_len': len(tokens[bs_i]),
                    #                 } for bs_i in range(dataloader.batch_size)]
                    #     generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        } for bs_i in range(dataloader.batch_size)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        # self.w_vectorizer = dataloader.dataset.w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, length, text= data['motion'], data['length'], data['text']
        # sent_len = data['cap_len']

        # if self.dataset.mode == 'eval':
        #     normed_motion = motion
        #     denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
        #     renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
        #     motion = renormed_motion
        #     # This step is needed because T2M evaluators expect their norm convention

        # pos_one_hots = []
        # word_embeddings = []
        # for token in tokens:
        #     word_emb, pos_oh = self.w_vectorizer[token]
        #     pos_one_hots.append(pos_oh[None, :])
        #     word_embeddings.append(word_emb[None, :])
        # pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        # word_embeddings = np.concatenate(word_embeddings, axis=0)
        return motion, length, text
        # return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

class CompTEACHGeneratedDataset(Dataset):

    def __init__(self, args, cnt):
        batch_size = 32
        print(f'loading generated_{cnt}.npy')
        self.generated_motion = []
        self.data = np.load(f'/home/zhao_yang/project/teach/data/generated_{cnt}.npy', allow_pickle=True)
        def collate_tensor_with_padding(batch):
            batch = [torch.tensor(x) for x in batch]
            dims = batch[0].dim()
            max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
            size = (len(batch),) + tuple(max_size)
            canvas = batch[0].new_zeros(size=size)
            for i, b in enumerate(batch):
                sub_tensor = canvas[i]
                for d in range(dims):
                    sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
                sub_tensor.add_(b)
            canvas = canvas.detach().numpy()
            # canvas = [x.detach().numpy() for x in canvas]
            return canvas
        # self.w_vectorizer = dataloader.dataset.w_vectorizer
        i = 0
        while i < 1024:
            motion_lst = []
            len_lst = []
            text_lst = []
            for j in range(i, i + 32):
                motion_lst.append(self.data[j]['motion'])
                len_lst.append(self.data[j]['length'])
                text_lst.append(self.data[j]['text'])

            motion_arr = collate_tensor_with_padding(motion_lst)
            sub_dicts = [{'motion': motion_arr[bs_i],
                'length': len_lst[bs_i],
                'text': text_lst[bs_i]
                # 'caption': model_kwargs['y']['text'][bs_i],
                # 'tokens': tokens[bs_i],
                # 'cap_len': len(tokens[bs_i]),
                } for bs_i in range(batch_size)]
            
            self.generated_motion += sub_dicts
            i += 32

        self.mm_generated_motion = mm_generated_motions = []

    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, length, text= data['motion'], data['length'], data['text']
        # sent_len = data['cap_len']

        # if self.dataset.mode == 'eval':
        #     normed_motion = motion
        #     denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
        #     renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
        #     motion = renormed_motion
        #     # This step is needed because T2M evaluators expect their norm convention

        # pos_one_hots = []
        # word_embeddings = []
        # for token in tokens:
        #     word_emb, pos_oh = self.w_vectorizer[token]
        #     pos_one_hots.append(pos_oh[None, :])
        #     word_embeddings.append(word_emb[None, :])
        # pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        # word_embeddings = np.concatenate(word_embeddings, axis=0)
        return motion, length, text
        # return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)