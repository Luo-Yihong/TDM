#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
from copy import deepcopy
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

# from unet import UNet2DConditionModel
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, LCMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

import torch as th
from torch import nn
import math

def generate_new(unet,noise_scheduler,latent, noise,encoder_hidden_states, prompt_attention_mask, 
                steps = 1, return_mid = False, mid_points = None, total_steps = 800, add_cfg = None):
    if add_cfg is not None:
        uncond_attention_mask = add_cfg['uncond_attention_mask']
        uncond_prompt_embeds = add_cfg['uncond_prompt_embeds']
        cfg = add_cfg['cfg']
    T_ = torch.randint(total_steps-1, total_steps, (latent.shape[0],), device=latent.device)
    T_ = T_.long()
    zero_t = torch.zeros_like(T_)
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod).to(latent.device).to(latent.dtype)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod).to(latent.device).to(latent.dtype)
    imgs_list = []
    # pure_noisy = noise_scheduler.add_noise(torch.randn_like(noise), noise, T_)
    pure_noisy = noise
    noisy_imgs_list = []
    added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
    for ind in range(steps):
        noisy_imgs_list.append(pure_noisy)
        noise_pred = unet(pure_noisy, timestep = T_, added_cond_kwargs = added_cond_kwargs,
                                            encoder_hidden_states = encoder_hidden_states, encoder_attention_mask=prompt_attention_mask, return_dict=False)[0]
        noise_pred = noise_pred.chunk(2, dim=1)[0]
        if add_cfg is not None:
            noise_pred_uncond = unet(pure_noisy, timestep = T_, added_cond_kwargs = added_cond_kwargs,
                                                encoder_hidden_states = uncond_prompt_embeds, encoder_attention_mask=uncond_attention_mask, return_dict=False)[0]
            noise_pred_uncond = noise_pred_uncond.chunk(2, dim=1)[0] 
            noise_pred = noise_pred_uncond + cfg * (noise_pred - noise_pred_uncond)
        latent = predicted_origin(  noise_pred,
                                    T_,
                                    pure_noisy,
                                    noise_scheduler.config.prediction_type,
                                    alpha_schedule,
                                    sigma_schedule,
                                )
        imgs_list.append(latent)
        if mid_points is not None:
            T_ = mid_points[ind+1] + zero_t
        else:
            T_ = T_ - total_steps // steps
        pure_noisy = noise_scheduler.add_noise(latent, noise_pred, T_).to(torch.float16)
    noisy_imgs_list.append(latent)
    if return_mid:
        return imgs_list, noisy_imgs_list
    return latent

def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def predicted_origin(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def get_module_kohya_state_dict(module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"):
    kohya_ss_state_dict = {}
    for peft_key, weight in get_peft_model_state_dict(module, adapter_name=adapter_name).items():
        if "base_model.model" in peft_key:
            kohya_key = peft_key.replace("base_model.model", prefix)
        else:
            kohya_key = prefix + '.' + peft_key
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)

        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(module.peft_config[adapter_name].lora_alpha).to(dtype)

    return kohya_ss_state_dict


def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        scheduler=LCMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
    args.seed = 42
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = []
    args.validation_prompts = [
        "realism, realistic, medieval, fantasy, masterwork thieves tools, lock picks, picks, small file, small mirror",
        "goo goo much plate, cutlery and water glass"
    ]
    for i in range(len(args.validation_prompts)):
        with torch.autocast("cuda"):
            image = pipeline(args.validation_prompts[i], num_inference_steps=1, generator=generator).images[0]
        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="PixArt-alpha/PixArt-XL-2-512x512",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="TDM-pixart",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
             "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=800,
        help="The weight for consistency loss."
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=2,
        help="The weight for consistency loss."
    )
    parser.add_argument(
        "--lambda_con",
        type=float,
        default=2,
        help="The weight for consistency loss."
    )
    parser.add_argument(
        "--lambda_kl",
        type=float,
        default=0.,
        help="The weight for KL loss",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--use_reg", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--use_huber", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--use_randmid", action="store_true", help="Whether or not to use randmid."
    )
    parser.add_argument(
        "--use_separate", action="store_true", help="Whether or not to use separate diffusing."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def main():
    args = parse_args()
    args.output_dir = args.output_dir + f"_cfg{args.cfg}_totalstep{args.total_steps}"
    if args.use_reg:
        args.output_dir = args.output_dir + "-Reg"
    if args.use_huber:
        args.output_dir = args.output_dir + "-Huber"
    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    import torch
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.

    noise_scheduler = DDPMScheduler(beta_start = 0.0001, beta_end =  0.02, beta_schedule = "linear",
                                    steps_offset = 1,trained_betas = None, clip_sample = False, rescale_betas_zero_snr = False)
    noise_scheduler.config.prediction_type = 'epsilon'
    # noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    alpha_schedule = alpha_schedule.to(accelerator.device).to(torch.float16)
    sigma_schedule = sigma_schedule.to(accelerator.device).to(torch.float16)

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    from transformers import T5EncoderModel, T5Tokenizer
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype = torch.float16
        )
        tokenizer = T5Tokenizer.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-512x512", subfolder="tokenizer",)
        from diffusers import AutoencoderTiny
        vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", )

    from diffusers import Transformer2DModel
    unet = Transformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )
    for param in unet.parameters(): 
        param.data = param.data.contiguous()

    unet_fake = Transformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )
    unet.enable_xformers_memory_efficient_attention()
    unet_fake.enable_xformers_memory_efficient_attention()
    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # unet.requires_grad_(False)
    unet_fake.requires_grad_(False)
    unet.train()
    unet_fake.train()
    from diffusers import AutoPipelineForText2Image

    from copy import deepcopy
    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = deepcopy(unet)
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=Transformer2DModel, model_config=ema_unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            unet_fake.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                unet_ = accelerator.unwrap_model(unet)
                unet_.save_pretrained(os.path.join(output_dir, "unet"))
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model
            unet_ = accelerator.unwrap_model(unet)

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        unet_fake.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    import torch
    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    

    unet.requires_grad_(True)
    lora_layers = [param for param in unet.parameters() if param.requires_grad]
                   
    unet_fake.requires_grad_(True)
    fakelora_layers = [param for param in unet_fake.parameters() if param.requires_grad]

    optimizer = optimizer_cls(
        lora_layers,
        lr=args.learning_rate / 5,
        betas=(0., 0.95),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    optimizer_d = optimizer_cls(
        fakelora_layers,
        lr=args.learning_rate,
        betas=(0., 0.95),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    import json
    import torch
    from torch.utils.data import Dataset
    from PIL import Image

    class CustomImagePromptDataset(Dataset):
        def __init__(self, jsonl_file, transform=None):
            self.data = []
            self.transform = transform
            i = 0
            pth_base = "/root/data/journey_prompt/realistic-vision"
            self.generator = torch.Generator()
            self.data = torch.load('./cache.pt',  weights_only=False)['data']
            self.tokenizer = T5Tokenizer.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-512x512", subfolder="tokenizer",)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            latents_pth, text, index = self.data[idx]
            text_input = self.tokenizer([text], max_length=120, padding="max_length", truncation=True, return_tensors="pt")#.input_ids
            return text, text_input.input_ids, text_input.attention_mask

    # Create Dataset
    dataset = CustomImagePromptDataset(jsonl_file='../yoso_release/train_anno.jsonl', transform=None)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=8,
        pin_memory=True,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # unet_gan.unet = accelerator.prepare_model(unet_gan.unet,find_unused_parameters = True)
    # unet_gan.out_head = accelerator.prepare_model(unet_gan.out_head)
    unet_fake = accelerator.prepare_model(unet_fake)

    # Prepare everything with our `accelerator`.
    unet, optimizer, optimizer_d, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, optimizer_d, train_dataloader, lr_scheduler  # , find_unused_parameters=True
    )
    if args.use_ema:
        print('EMA!!!!!!!!!!!!!!')
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    from copy import deepcopy
    # unet_sd = deepcopy(unet)
    unet_sd = Transformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )
    unet_sd.eval()
    unet_sd.requires_grad_(False)
    unet_sd.enable_xformers_memory_efficient_attention()
    unet_sd.requires_grad_(False)
    unet_sd = accelerator.prepare_model(unet_sd)
    unet_sd.requires_grad_(False)

    # Potentially load in the weights and states from a previous save
    print(args.resume_from_checkpoint)
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    from copy import deepcopy
    import torch.nn.functional as F

    with torch.no_grad():
        uncond_input = tokenizer(
            [""] * args.train_batch_size,
            return_tensors="pt",
            padding="max_length", max_length=120
        ).to(accelerator.device)
        uncond_attention_mask = uncond_input.attention_mask.to(torch.float16)
        uncond_prompt_embeds = text_encoder(uncond_input.input_ids, return_dict=False, attention_mask=uncond_input.attention_mask)[0]
        fixed_input = tokenizer(
            ["A photo of a cat", "A photo of a dog", "A photo of a panda", "A photo of a pikachu"],
            return_tensors="pt",
            padding="max_length", max_length=120
        ).to(accelerator.device)
        fixed_prompt_embeds = text_encoder(fixed_input.input_ids, return_dict=False, attention_mask=fixed_input.attention_mask)[0]
        fixed_mask = fixed_input.attention_mask.to(torch.float16)
        add_cfg = {"uncond_attention_mask": uncond_attention_mask[:fixed_mask.shape[0]], "uncond_prompt_embeds": uncond_prompt_embeds[:fixed_mask.shape[0]], 'cfg': 7.5}

    class Predictor():
        def __init__(self, noise_scheduler,
                     alpha_schedule,
                     sigma_schedule):
            super(Predictor, self).__init__()
            self.noise_scheduler = noise_scheduler
            self.alpha_schedule = alpha_schedule
            self.sigma_schedule = sigma_schedule
            self.uncond_prompt_embeds = uncond_prompt_embeds
            self.uncond_attention_mask = uncond_attention_mask
            self.added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        def predict(self, score_model, noisy_samples, timesteps, encoder_hidden_states, prompt_attention_mask, cfg=None, steps=1,
                    return_double=False, timestep_cond = None):
            for _ in range(steps):
                score_pred = score_model(noisy_samples, timestep = timesteps, added_cond_kwargs = self.added_cond_kwargs,
                                                    encoder_hidden_states = encoder_hidden_states, encoder_attention_mask=prompt_attention_mask, return_dict=False)[0]
                score_pred = score_pred.chunk(2, dim=1)[0]
                # score_pred = score_model(noisy_samples, timesteps, encoder_hidden_states, timestep_cond = timestep_cond, return_dict=False)[0]
                if cfg is not None:
                    score_uncon_pred = score_model(noisy_samples, timestep = timesteps, added_cond_kwargs = self.added_cond_kwargs,
                                    encoder_hidden_states = self.uncond_prompt_embeds, encoder_attention_mask=self.uncond_attention_mask, return_dict=False)[0]
                    score_uncon_pred = score_uncon_pred.chunk(2, dim=1)[0]
                    # score_pred_cfg = score_pred + cfg * (score_pred - score_uncon_pred)
                    score_pred_cfg = score_uncon_pred + cfg * (score_pred - score_uncon_pred)
                    pred_latents = predicted_origin(
                        score_pred_cfg,
                        timesteps.long(),
                        noisy_samples,
                        self.noise_scheduler.config.prediction_type,
                        self.alpha_schedule,
                        self.sigma_schedule, )
                    pred_latents_nocfg = predicted_origin(
                        score_pred,
                        timesteps.long(),
                        noisy_samples,
                        self.noise_scheduler.config.prediction_type,
                        self.alpha_schedule,
                        self.sigma_schedule, )
                    timesteps = timesteps - timesteps // steps
                    noisy_samples = self.noise_scheduler.add_noise(pred_latents.detach(), score_pred_cfg, timesteps.long())
                    if return_double:
                        return score_pred_cfg, pred_latents, pred_latents_nocfg
                else:
                    pred_latents = predicted_origin(
                        score_pred,
                        timesteps,
                        noisy_samples,
                        self.noise_scheduler.config.prediction_type,
                        self.alpha_schedule,
                        self.sigma_schedule, )
                    timesteps = timesteps - timesteps // steps
                    noisy_samples = self.noise_scheduler.add_noise(pred_latents.detach(), score_pred, timesteps)
            if cfg is not None:
                return score_pred_cfg, pred_latents
            else:
                return score_pred, pred_latents
            
        def add_noise(self, samples, noise, t1, t2):
            sigmas = extract_into_tensor(self.sigma_schedule, t1, samples.shape)
            alphas = extract_into_tensor(self.alpha_schedule, t1, samples.shape)
            sigmas_new = extract_into_tensor(self.sigma_schedule, t2, samples.shape)
            alphas_new = extract_into_tensor(self.alpha_schedule, t2, samples.shape)
            samples = samples / alphas * alphas_new # x = alphas_new * x + alphas_new / alphas * sigmas * eps
            beta = sigmas_new ** 2 - ( alphas_new / alphas * sigmas ) ** 2
            beta = beta ** 0.5
            samples = samples + beta * noise
            return samples.to(torch.float16)
        
        def obtain_mixed_noise(self, model_noise, noise, t1, t2):
            sigmas = extract_into_tensor(self.sigma_schedule, t1, model_noise.shape)
            alphas = extract_into_tensor(self.alpha_schedule, t1, model_noise.shape)
            sigmas_new = extract_into_tensor(self.sigma_schedule, t2, model_noise.shape)
            alphas_new = extract_into_tensor(self.alpha_schedule, t2, model_noise.shape)
            
            beta = sigmas_new ** 2 - ( alphas_new / alphas * sigmas ) ** 2
            beta = beta ** 0.5
            mixed_noise = model_noise / alphas * alphas_new * sigmas + beta * noise
            mixed_noise = mixed_noise / sigmas_new
            return mixed_noise
            
    predictor = Predictor(noise_scheduler, alpha_schedule, sigma_schedule)

    fixed_noise = None
    fixed_c = None
    total_steps = args.total_steps
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0    
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, unet_fake):
                text_ = list(batch[0])
                noise = torch.randn([len(text_), 4, 64, 64]).to(torch.float16).to(accelerator.device)
                latents = noise


                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
    
                # noise = torch.randn_like(latents)
                new_noise = torch.randn_like(noise)
                bsz = noise.shape[0]
                # Sample a random timestep for each image
                T = torch.randint(total_steps - 1, total_steps, (bsz,), device=noise.device).long()
                # Get the text embedding for conditioning
                # input_ids = batch[1]  # .to(weight_dtype)
                with torch.no_grad():
                    input_ids = batch[1].view(batch[1].shape[0],-1)#.input_ids#.to(weight_dtype)
                    prompt_attention_mask = batch[2].view(batch[1].shape[0],-1)#.attention_mask
                    encoder_hidden_states = text_encoder(input_ids, return_dict=False, attention_mask=prompt_attention_mask)[0]
                    
                
                with torch.no_grad():
                    imgs_list, noisy_imgs_list = generate_new(unet,noise_scheduler,noise, noise,encoder_hidden_states, prompt_attention_mask, steps = 4, return_mid = True, total_steps = args.total_steps) # [ [bs,4,64,64] * K ]
                    noisy_imgs_list.reverse()
                
                fw_t = 240
                # Train the Fake score model
                unet_fake.requires_grad_(True)
                with torch.no_grad():
                    encoder_hidden_states_fake = encoder_hidden_states
                    ind_t = torch.randint(1, 5, (bsz,),
                                              device=noise.device).long()
                    noisy_latents = torch.randn_like(latents)
                    for i in range(latents.shape[0]):
                        noisy_latents[i] = noisy_imgs_list[ind_t[i]][i]
                    noise_g = torch.randn_like(latents)
                    timesteps_g = ind_t * total_steps // 4 - 1
                    timesteps_mid_ori = timesteps_g -  total_steps // 4 + 1
                    timesteps_mid = timesteps_mid_ori.clone()
                    if args.use_randmid: # This can regularize the generator.
                        for i in range(latents.shape[0]):
                            timesteps_mid[i] = torch.randint(timesteps_mid_ori[i], timesteps_g[i]-20, (1,),device=noise.device)[0].long()
                    timesteps = timesteps_g * 0 
                    for i in range(latents.shape[0]):
                        lowt = max(20, timesteps_mid[i])
                        if args.use_separate:
                            upt = timesteps_g[i] - 10
                        else:
                            upt = args.total_steps - 10
                        timesteps[i] = torch.randint(lowt, upt, (1,),device=noise.device)[0].long()
                    model_eps, model_latents = predictor.predict(unet, noisy_latents, timesteps_g, encoder_hidden_states, prompt_attention_mask)
                    noisy_model_latents_ode = noise_scheduler.add_noise(model_latents, model_eps, timesteps_mid).to(torch.float16)
                    rand_noise = torch.randn_like(noisy_model_latents_ode)
                    noisy_model_latents = predictor.add_noise(noisy_model_latents_ode.detach(), rand_noise, timesteps_mid, timesteps).to(torch.float16)
                    mixed_noise = predictor.obtain_mixed_noise(model_eps, rand_noise, timesteps_mid, timesteps)
                    sd_eps, sd_latents = predictor.predict(unet_sd, noisy_model_latents, timesteps,
                                                           encoder_hidden_states_fake, prompt_attention_mask, cfg=None, steps=1)
                fake_pred, fake_latents = predictor.predict(unet_fake, noisy_model_latents, timesteps,
                                                            encoder_hidden_states_fake, prompt_attention_mask)
                is_weight = torch.exp( - 0.5 * ((mixed_noise) ** 2).view(bsz,-1).mean(dim=1) ) / torch.exp( -  0.5 *((rand_noise) ** 2).view(bsz,-1).mean(dim=1) )
                loss_score = F.mse_loss(fake_latents.float(), model_latents.float(), reduction="none")
                snr = compute_snr(noise_scheduler, timesteps)
                snr = torch.stack([snr, 5 * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] 
                loss_score = loss_score.mean(dim=list(range(1, len(loss_score.shape)))) * snr * is_weight
                loss_score = loss_score.mean()
                accelerator.backward(loss_score)
                optimizer_d.step()
                optimizer_d.zero_grad()

                # Train the Few-Step Unet generator
                if global_step % 1 == 0:
                    ind_t = torch.randint(1, 5, (bsz,),
                                              device=noise.device).long()
                    noisy_latents = torch.randn_like(latents)
                    for i in range(latents.shape[0]):
                        noisy_latents[i] = noisy_imgs_list[ind_t[i]][i]
                    noise_g = torch.randn_like(latents)
                    timesteps_g = ind_t * total_steps // 4 - 1
                    timesteps_mid = timesteps_mid_ori.clone()
                    if args.use_randmid: # This can regularize the generator.
                        for i in range(latents.shape[0]):
                            timesteps_mid[i] = torch.randint(timesteps_mid_ori[i], timesteps_g[i]-20, (1,),device=noise.device)[0].long()
                    timesteps = timesteps_g * 0 
                    for i in range(latents.shape[0]):
                        lowt = max(20, timesteps_mid[i])
                        if args.use_separate:
                            upt = timesteps_g[i] - 10
                        else:
                            upt = args.total_steps - 10
                        timesteps[i] = torch.randint(lowt, upt, (1,),device=noise.device)[0].long()
                    model_eps, model_latents = predictor.predict(unet, noisy_latents, timesteps_g, encoder_hidden_states, prompt_attention_mask)
                    noisy_model_latents_ode = noise_scheduler.add_noise(model_latents, model_eps, timesteps_mid).to(torch.float16)
                    noisy_model_latents = predictor.add_noise(noisy_model_latents_ode, torch.randn_like(noisy_model_latents_ode), timesteps_mid, timesteps).to(torch.float16)
                    snr = compute_snr(noise_scheduler, timesteps)
                    coop_samples = model_latents.detach().clone()
                    cfg = args.cfg
                    with torch.no_grad():
                        sd_eps, sd_latents = predictor.predict(unet_sd, noisy_model_latents, timesteps,
                                                                                encoder_hidden_states, prompt_attention_mask, cfg=None, steps=1)
                        sd_eps_uncond, sd_latents_uncond = predictor.predict(unet_sd, noisy_model_latents, timesteps,
                                                                                uncond_prompt_embeds, uncond_attention_mask, cfg=None, steps=1)
                        _, fake_latents = predictor.predict(unet_fake, noisy_model_latents, timesteps,
                                                                    encoder_hidden_states, prompt_attention_mask, steps=1, cfg=None)
                        coop_samples = coop_samples.detach().clone() + 1 * (
                            sd_latents - fake_latents).detach().clone() + (cfg-1) * (
                                            sd_latents - sd_latents_uncond).detach().clone()
                    sd_latents_cfg = sd_latents + (cfg-1) * (sd_latents - sd_latents_uncond)
                    weighting_factor = torch.abs(model_latents.double() - sd_latents_cfg.double() ).mean(dim=[1, 2, 3], keepdim=True).detach()
                    if args.use_huber:
                        args.huber_c = 1e-3
                        loss_instruct = torch.mean(
                        (torch.sqrt((model_latents.float() - coop_samples.detach().float()) ** 2 + args.huber_c**2) - args.huber_c) / weighting_factor
                    )
                    else:
                        loss_instruct = F.mse_loss(model_latents.float(), coop_samples.detach().float(),
                                                reduction='none') / weighting_factor
                        loss_instruct = loss_instruct.mean()
                    loss = loss_instruct
                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                if accelerator.sync_gradients:
                    if global_step % 50 == 0:
                        if fixed_c is None:
                            fixed_c = fixed_prompt_embeds.to(torch.float16).to(accelerator.device)
                            fixed_mask = fixed_mask.to(torch.float16).to(accelerator.device)
                            fixed_noise = torch.randn([4, 4, 64, 64]).to(torch.float16).to(accelerator.device)
                            fixed_T = T
                        with torch.no_grad():
                            fixed_latents = generate_new(unet,noise_scheduler,fixed_noise, fixed_noise,fixed_c, fixed_mask, steps = 4, total_steps = args.total_steps)
                            fixed_latents_1step = generate_new(unet,noise_scheduler,fixed_noise, fixed_noise,fixed_c, fixed_mask, steps = 1, total_steps = args.total_steps)
                            images_noise = vae.decode(fixed_latents[:4].to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
                            images_fixed1 = vae.decode(fixed_latents_1step[:4].to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0].clamp(-1, 1) * 0.5 + 0.5
                            latent_4step = generate_new(unet,noise_scheduler,latents, noise,encoder_hidden_states, prompt_attention_mask, steps = 4, total_steps = args.total_steps)
                            images_4step = vae.decode(latent_4step.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0].clamp(-1,1)*0.5+0.5
                            images_1step = \
                                vae.decode(model_latents[:4].to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[
                                    0]
                            images_1step = images_1step.clamp(-1, 1) * 0.5 + 0.5
                            images_noise = images_noise.clamp(-1, 1) * 0.5 + 0.5                    
                        if accelerator.is_main_process:
                            save_image(images_1step, f'./{args.output_dir}/1step.jpg', normalize=False, nrow=2)
                            save_image(images_noise, f'./{args.output_dir}/fixed_4step_{global_step}.jpg', normalize=False, nrow=2)
                            save_image(images_fixed1, f'./{args.output_dir}/fixed_1step_{global_step}.jpg', normalize=False, nrow=2)
                            save_image(images_4step[:4], f'./{args.output_dir}/4step.jpg', normalize = False, nrow = 2)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"loss_instruct": loss_instruct, },
                                step=global_step)
                train_loss = 0.0
                train_d_real = 0.0
                train_d_fake = 0.0

                if global_step % (args.checkpointing_steps) == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss_score": loss_score.detach().item(), "loss_instruct": loss_instruct.detach().item(), }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
    accelerator.end_training()


if __name__ == "__main__":
    main()
