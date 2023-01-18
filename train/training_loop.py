import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from eval import eval_humanml, eval_humanact12_uestc
from data_loaders.get_data import get_dataset_loader

from teach.data.tools import lengths_to_mask    
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1
        print(f"Total Epochs: {self.num_epochs}")
        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        if args.dataset in ['kit', 'humanml'] and args.eval_during_training:
            mm_num_samples = 0  # mm is super slow hence we won't run it during training
            mm_num_repeats = 0  # mm is super slow hence we won't run it during training
            gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                            split=args.eval_split,
                                            hml_mode='eval')

            self.eval_gt_data = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                                   split=args.eval_split,
                                                   hml_mode='gt')
            self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
            self.eval_data = {
                'test': lambda: eval_humanml.get_mdm_loader(
                    model, diffusion, args.eval_batch_size,
                    gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
                    args.eval_num_samples, scale=1.,
                )
            }
        self.use_ddp = False
        self.ddp_model = self.model

        self.hist_frames = args.hist_frames
        self.inpainting_frames = args.inpainting_frames

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            missing_keys, unexpected_keys = self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                ),
                strict=False
            )
            assert len(unexpected_keys) == 0
            assert all([k.startswith('clip_model.') for k in missing_keys])

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):

        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            if self.args.dataset == 'babel':
                for batch in tqdm(self.data):
                    if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                        break
                    batch = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in batch.items()}
                    if self.args.arch == 'past_cond':
                        self.run_step_multi(batch)
                    else:
                        self.run_step_inpainting(batch)
                    if self.step % self.log_interval == 0:
                        for k,v in logger.get_current().name2val.items():
                            if k == 'loss':
                                print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))

                            if k in ['step', 'samples'] or '_q' in k:
                                continue
                            else:
                                self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')

                    if self.step % self.save_interval == 0:
                        self.save()
                        self.model.eval()
                        self.evaluate()
                        self.model.train()

                        # Run for a finite amount of time in integration tests.
                        if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                            return
                    self.step += 1
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break
            else:
                for motion, cond in tqdm(self.data):
                    if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                        break

                    motion = motion.to(self.device)
                    cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

                    self.run_step(motion, cond)
                    if self.step % self.log_interval == 0:
                        for k,v in logger.get_current().name2val.items():
                            if k == 'loss':
                                print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))

                            if k in ['step', 'samples'] or '_q' in k:
                                continue
                            else:
                                self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')

                    if self.step % self.save_interval == 0:
                        self.save()
                        self.model.eval()
                        self.evaluate()
                        self.model.train()

                        # Run for a finite amount of time in integration tests.
                        if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                            return
                    self.step += 1
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def evaluate(self):
        if not self.args.eval_during_training:
            return
        start_eval = time.time()
        if self.eval_wrapper is not None:
            print('Running evaluation loop: [Should take about 90 min]')
            log_file = os.path.join(self.save_dir, f'eval_humanml_{(self.step + self.resume_step):09d}.log')
            diversity_times = 300
            mm_num_times = 0  # mm is super slow hence we won't run it during training
            eval_dict = eval_humanml.evaluation(
                self.eval_wrapper, self.eval_gt_data, self.eval_data, log_file,
                replication_times=self.args.eval_rep_times, diversity_times=diversity_times, mm_num_times=mm_num_times, run_mm=False)
            print(eval_dict)
            for k, v in eval_dict.items():
                if k.startswith('R_precision'):
                    for i in range(len(v)):
                        self.train_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
                                                          iteration=self.step + self.resume_step,
                                                          group_name='Eval')
                else:
                    self.train_platform.report_scalar(name=k, value=v, iteration=self.step + self.resume_step,
                                                      group_name='Eval')

        elif self.dataset in ['humanact12', 'uestc']:
            eval_args = SimpleNamespace(num_seeds=self.args.eval_rep_times, num_samples=self.args.eval_num_samples,
                                        batch_size=self.args.eval_batch_size, device=self.device, guidance_param = 1,
                                        dataset=self.dataset, unconstrained=self.args.unconstrained,
                                        model_path=os.path.join(self.save_dir, self.ckpt_file_name()))
            eval_dict = eval_humanact12_uestc.evaluate(eval_args, model=self.model, diffusion=self.diffusion, data=self.data.dataset)
            print(f'Evaluation results on {self.dataset}: {sorted(eval_dict["feats"].items())}')
            for k, v in eval_dict["feats"].items():
                if 'unconstrained' not in k:
                    self.train_platform.report_scalar(name=k, value=np.array(v).astype(float).mean(), iteration=self.step, group_name='Eval')
                else:
                    self.train_platform.report_scalar(name=k, value=np.array(v).astype(float).mean(), iteration=self.step, group_name='Eval Unconstrained')

        end_eval = time.time()
        print(f'Evaluation time: {round(end_eval-start_eval)/60}min')


    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def run_step_multi(self, batch):
        self.forward_backward_multi(batch)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def run_step_inpainting(self, batch):
        self.forward_backward_inpainting(batch)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset=self.data.dataset
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def forward_backward_multi(self, batch):
        self.mp_trainer.zero_grad()
        for i in range(0, batch['motion_feats_0'].shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro_0 = batch['motion_feats_0']
            micro_0 = micro_0.unsqueeze(2).permute(0, 3, 2, 1)
            micro_cond_0 = {}
            micro_cond_0['y'] = {}
            micro_cond_0['y']['length'] = batch['length_0']
            # assuming mask.shape == bs, 1, 1, seqlen
            micro_cond_0['y']['mask'] = lengths_to_mask(batch['length_0'], micro_0.device).unsqueeze(1).unsqueeze(2)
            micro_cond_0['y']['text'] = batch['text_0']
            
            last_batch = (i + self.microbatch) >= batch['motion_feats_0'].shape[0]
            t, weights = self.schedule_sampler.sample(micro_0.shape[0], dist_util.dev())
            compute_losses0 = functools.partial(
                self.diffusion.training_losses_multi,
                self.ddp_model,
                micro_0,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond_0,
                dataset=self.data.dataset
            )
            if last_batch or not self.use_ddp:
                # hist_frames [b 5 dim]
                loss0, hist = compute_losses0() 
            else:
                with self.ddp_model.no_sync():
                    loss0, hist = compute_losses0() 

            # bs 135 1 frames
            # bs = hist.shape(0)
            if self.hist_frames > 0:
                hist_lst = [feats[:,:,:len] for feats, len in zip(hist, batch['length_0'])]
                hframes = torch.stack([x[:,:,-self.hist_frames:] for x in hist_lst])
                # micro_cond_1['y']['hframes'] = hframes

            micro_1 = batch['motion_feats_1_with_transition']
            micro_1 = micro_1.unsqueeze(2).permute(0, 3, 2, 1)
            micro_cond_1 = {}
            micro_cond_1['y'] = {}
            if self.hist_frames > 0:
                micro_cond_1['y']['hframes'] = hframes
            micro_cond_1['y']['length'] = batch['length_1_with_transition']
            micro_cond_1['y']['mask'] = lengths_to_mask(batch['length_1_with_transition'], micro_1.device).unsqueeze(1).unsqueeze(2)
            micro_cond_1['y']['text'] = batch['text_1']
            
            # micro_cond = cond
            
            #t, weights = self.schedule_sampler.sample(micro_0.shape[0], dist_util.dev())
            compute_losses1 = functools.partial(
                self.diffusion.training_losses_multi,
                self.ddp_model,
                micro_1,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond_1,
                dataset=self.data.dataset
            )
            if last_batch or not self.use_ddp:
                # hist_frames [b 5 dim]
                loss1, _ = compute_losses1() 
            else:
                with self.ddp_model.no_sync():
                    loss1, _ = compute_losses1() 
            # print_0 = (loss0['loss']).mean()
            # print_1 = (loss1['loss']).mean()
            # print(f'loss_0: {print_0}, loss_1:{print_1}')
            losses = {}
            losses['loss'] = loss0['loss'] + loss1['loss']
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def forward_backward_inpainting(self, batch):
        self.mp_trainer.zero_grad()
        for i in range(0, batch['motion_feats_0'].shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            last_batch = (i + self.microbatch) >= batch['motion_feats_0'].shape[0]
            t, weights = self.schedule_sampler.sample(batch['motion_feats_0'].shape[0], dist_util.dev())

            micro_0 = batch['motion_feats_0'] # bs len 135
            micro_0 = micro_0.unsqueeze(2).permute(0, 3, 2, 1) # bs 135 1 len
            micro_cond_0 = {}
            micro_cond_0['y'] = {}
            micro_cond_0['y']['length'] = batch['length_0']
            # assuming mask.shape == bs, 1, 1, seqlen
            micro_cond_0['y']['mask'] = lengths_to_mask(batch['length_0'], micro_0.device).unsqueeze(1).unsqueeze(2)
            micro_cond_0['y']['text'] = batch['text_0']
            
            compute_losses_0 = functools.partial(
                self.diffusion.training_losses_inpainting,
                self.ddp_model,
                micro_0,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond_0,
                dataset=self.data.dataset
            )
            if last_batch or not self.use_ddp:
                # hist_frames [b 5 dim]
                loss_0 = compute_losses_0() 
            else:
                with self.ddp_model.no_sync():
                    loss_0 = compute_losses_0() 

            # bs 135 1 frames

            micro_1 = batch['motion_feats_1_with_transition']
            micro_1 = micro_1.unsqueeze(2).permute(0, 3, 2, 1)
            micro_cond_1 = {}
            micro_cond_1['y'] = {}
            micro_cond_1['y']['length'] = [self.inpainting_frames + len for len in batch['length_1_with_transition']]
            micro_cond_1['y']['mask'] = lengths_to_mask(micro_cond_1['y']['length'], micro_1.device).unsqueeze(1).unsqueeze(2)
            micro_cond_1['y']['text'] = batch['text_1']
            hist_lst = [feats[:,:,:len] for feats, len in zip(micro_0, micro_cond_0['y']['length'])]
            hist_frames = torch.stack([x[:,:,-self.inpainting_frames:] for x in hist_lst])
            micro_1 = torch.cat((hist_frames, micro_1), axis=-1)
            # micro_cond = cond
            
            #t, weights = self.schedule_sampler.sample(micro_0.shape[0], dist_util.dev())
            compute_losses_1 = functools.partial(
                self.diffusion.training_losses_inpainting,
                self.ddp_model,
                micro_1,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond_1,
                dataset=self.data.dataset
            )
            if last_batch or not self.use_ddp:
                # hist_frames [b 5 dim]
                loss_1 = compute_losses_1() 
            else:
                with self.ddp_model.no_sync():
                    loss_1 = compute_losses_1() 
            # print_0 = (loss0['loss']).mean()
            # print_1 = (loss1['loss']).mean()
            # print(f'loss_0: {print_0}, loss_1:{print_1}')

            micro_2 = batch['motion_feats_0']
            micro_2 = micro_2.unsqueeze(2).permute(0, 3, 2, 1)
            micro_cond_2 = {}
            micro_cond_2['y'] = {}
            micro_cond_2['y']['length'] = [len + self.inpainting_frames for len in batch['length_0']]
            micro_cond_2['y']['mask'] = lengths_to_mask(micro_cond_2['y']['length'], micro_2.device).unsqueeze(1).unsqueeze(2)
            micro_cond_2['y']['text'] = batch['text_0']
            fut_lst = [feats[:,:,:len] for feats, len in zip(batch['motion_feats_1_with_transition'].unsqueeze(2).permute(0, 3, 2, 1), batch['length_1_with_transition'])]
            fut_frames = torch.stack([x[:,:,:self.inpainting_frames] for x in fut_lst])
            micro_2 = torch.cat((micro_2, torch.zeros(micro_2.shape[0], micro_2.shape[1], micro_2.shape[2], self.inpainting_frames).to(micro_2.device)), axis=-1)
            for idx in range(micro_2.shape[0]):
                micro_2[idx, :, :, batch['length_0'][idx]:batch['length_0'][idx]+self.inpainting_frames] = fut_frames[idx, :, :, :] 
            # micro_2 = torch.cat((micro_2, fut_frames), axis=-1)
            # micro_cond = cond
            
            #t, weights = self.schedule_sampler.sample(micro_0.shape[0], dist_util.dev())
            compute_losses_cycle = functools.partial(
                self.diffusion.training_losses_inpainting,
                self.ddp_model,
                micro_2,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond_2,
                dataset=self.data.dataset
            )
            if last_batch or not self.use_ddp:
                # hist_frames [b 5 dim]
                loss_cycle = compute_losses_cycle() 
            else:
                with self.ddp_model.no_sync():
                    loss_cycle = compute_losses_cycle() 

            losses = {}
            losses['loss'] = loss_0['loss'] + loss_1['loss'] + self.args.lambda_cycle * loss_cycle['loss']
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)


    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"


    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
