# File based on https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

import argparse
import itertools
import logging
import os
import shutil
import numpy as np

import random
import accelerate

import diffusers
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.checkpoint
import torchvision
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoProcessor
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers import UniPCMultistepScheduler


from ootd.pipelines_ootd.pipeline_ootd import OotdPipeline

from ootd.pipelines_ootd.unet_garm_2d_condition import UNetGarm2DConditionModel
from ootd.pipelines_ootd.unet_vton_2d_condition import UNetVton2DConditionModel
from ootd.inference_ootd import VDPDiffusion

from dataset import VDPDataset
#from val_metrics import generate_images_from_vdp_pipe

#from val_metrics import compute_metrics

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_START_METHOD"] = "thread"

torch.multiprocessing.set_sharing_strategy('file_system')

from bitsandbytes.optim import AdamW8bit

def parse_args():
    parser = argparse.ArgumentParser(description="VTO training script.")
    parser.add_argument("--trainset-path", type=str, required=True, help="train dataset")
    parser.add_argument("--testset-path", type=str, required=True, help="test dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )


    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")

    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )

    parser.add_argument(
        "--test_batch_size", type=int, default=16, help="Batch size (per device) for the testing dataloader."
    )

    parser.add_argument("--num_train_epochs", type=int, default=100, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=200001,
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
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='fp16',
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
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=50000,
        help=(
            "Perform validation step and save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to use in the dataloaders.")
    parser.add_argument("--num_workers_test", type=int, default=8,
                        help="Number of workers to use in the test dataloaders.")

    parser.add_argument("--test_order", type=str, default="unpaired", choices=["unpaired", "paired"])
    parser.add_argument("--uncond_fraction", type=float, default=0.2, help="Fraction of unconditioned training samples")

    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit AdamW optimizer.")

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


VIT_PATH = "openai/clip-vit-large-patch14"
VAE_PATH = "CompVis/stable-diffusion-v1-4"
UNET_PATH = "CompVis/stable-diffusion-v1-4"
MODEL_PATH = "CompVis/stable-diffusion-v1-4"


def main():
    args = parse_args()

    # Setup accelerator.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    vae = AutoencoderKL.from_pretrained(
        VAE_PATH,
        subfolder="vae",
        torch_dtype=torch.float16,
    )

    unet_garm = UNetGarm2DConditionModel.from_pretrained(
        UNET_PATH,
        subfolder="unet",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    unet_vton = UNetVton2DConditionModel.from_pretrained(
        UNET_PATH,
        subfolder="unet",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    
    auto_processor = AutoProcessor.from_pretrained(VIT_PATH)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH).to(accelerator.device, dtype=weight_dtype)

    tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_PATH,
            subfolder="tokenizer",
    )

    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_PATH,
        subfolder="text_encoder",
    ).to(accelerator.device, dtype=weight_dtype)


    ##### ?????????

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    #auto_processor.requires_grad_(False)
    image_encoder.requires_grad_(False)
    #image_processor.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet_vton.enable_xformers_memory_efficient_attention()
            unet_garm.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        # Enable gradient checkpointing for memory efficient training
        unet_garm.enable_gradient_checkpointing()
        unet_vton.enable_gradient_checkpointing()


    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    params_to_optimize = list(unet_garm.parameters()) + list(unet_vton.parameters())
    if args.use_8bit_adam:
        optimizer = AdamW8bit(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )


    train_dataset = VDPDataset(
        path=args.trainset_path,
        image_processor=image_processor,
        auto_processor=auto_processor,
        size=(16, 16),
    )

    test_dataset = VDPDataset(
        path=args.testset_path,
        image_processor=image_processor,
        auto_processor=auto_processor,
        size=(16, 16),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers_test,
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
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    unet_garm, unet_vton,  optimizer, train_dataloader, lr_scheduler, test_dataloader = accelerator.prepare(
        unet_garm, unet_vton, optimizer, train_dataloader, lr_scheduler, test_dataloader
    )
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    # Move and vae to gpu and cast to weight_dtype
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
        accelerator.init_trackers("VDP_diff", config=vars(args),
                                  init_kwargs={"wandb": {"name": os.path.basename(args.output_dir)}})
        if args.report_to == 'wandb':
            wandb_tracker = accelerator.get_tracker("wandb")
            wandb_tracker.name = os.path.basename(args.output_dir)
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        try:
            if args.resume_from_checkpoint != "latest":
                path = os.path.basename(os.path.join("checkpoint", args.resume_from_checkpoint))
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(os.path.join(args.output_dir, "checkpoint"))
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1]
            accelerator.print(f"Resuming from checkpoint {path}")

            accelerator.load_state(os.path.join(args.output_dir, "checkpoint", path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch
        except Exception as e:
            print("Failed to load checkpoint, training from scratch:")
            print(e)
            resume_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet_garm.train()
        unet_vton.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet_garm), accelerator.accumulate(unet_vton):
                caption = batch["caption"]
                img_aug = batch["img_aug"].to(device=accelerator.device, dtype=weight_dtype)
                img = batch["img"].to(device=accelerator.device, dtype=weight_dtype)
                prompt_image = batch["prompt_img"].data['pixel_values'].to(device=accelerator.device, dtype=weight_dtype)

                prompt_image = image_encoder(prompt_image.squeeze(1)).image_embeds
                prompt_image = prompt_image.unsqueeze(1)

                ####################### a1
                prompt_embeds = tokenizer(caption, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
                ####################### a2
                prompt_embeds = text_encoder(prompt_embeds.input_ids.to(accelerator.device))[0]

                ####################### ?
                prompt = torch.cat([prompt_embeds, prompt_image], dim=1)

                # Set timesteps
                bsz = img.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=accelerator.device)
                timesteps = timesteps.long()

                seed = random.randint(0, 2147483647)
                generator = torch.manual_seed(seed)

                # 5. Prepare Image latents
                img_aug_latents = vae.encode(img_aug).latent_dist.mode()
                #img_aug_latents = img_aug_latents * vae.config.scaling_factor
                # TODO: multiply with scaling factor
                
                img_latents = vae.encode(img).latent_dist.mode()
                img_latents = img_latents * vae.config.scaling_factor

                #height, width = img_latents.shape[-2:]
                
                # We will denoise a single step, we dont start from pure noise, so we deleted latents in the pipeline

                _, spatial_attn_outputs = unet_garm(
                    img_aug_latents,
                    0,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )

                noise = torch.randn_like(img_latents)
                
                # unet_outfitting_encoder_hidden_states = encode_text_word_embedding(text_encoder, tokenized_text, cloth_clip, args.num_vstar).last_hidden_state
                denoising_unet_input = noise_scheduler.add_noise(img_latents, noise, timesteps)

                # predict the noise residual
                noise_pred = unet_vton(
                    denoising_unet_input,
                    spatial_attn_outputs,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

                # loss in accelerator.autocast according to docs https://huggingface.co/docs/accelerate/v0.15.0/quicktour#mixed-precision-training
                with accelerator.autocast():
                    loss = F.mse_loss(noise_pred, noise, reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(itertools.chain(unet_garm.parameters(), unet_vton.parameters()),
                                                args.max_grad_norm)


                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # Save checkpoint every checkpointing_steps steps
                # eval
                if global_step % args.checkpointing_steps == 0:
                    unet_garm.eval()
                    unet_vton.eval()

                    if accelerator.is_main_process:
                        os.makedirs(os.path.join(args.output_dir, "checkpoint"), exist_ok=True)
                        accelerator_state_path = os.path.join(args.output_dir, "checkpoint",
                                                              f"checkpoint-{global_step}")
                        accelerator.save_state(accelerator_state_path)

                        # Unwrap the Unet
                        unwrapped_unet_garm = accelerator.unwrap_model(unet_garm, keep_fp32_wrapper=True)
                        unwrapped_unet_vton = accelerator.unwrap_model(unet_vton, keep_fp32_wrapper=True)

                        with torch.no_grad():
                            val_pipe = OotdPipeline.from_pretrained(
                                MODEL_PATH,
                                unet_garm=unwrapped_unet_garm,
                                unet_vton=unwrapped_unet_vton,
                                vae=vae,
                                torch_dtype=torch.float16,
                                variant="fp16",
                                use_safetensors=True,
                                safety_checker=None,
                                requires_safety_checker=False,
                            ).to(accelerator.device)


                            val_pipe.scheduler = UniPCMultistepScheduler.from_config(val_pipe.scheduler.config)

                            # Generate a sample image for evaluation
                            img_given = next(iter(test_dataloader))['img_aug'].to(accelerator.device)
                            img_style = next(iter(test_dataloader))['img'].to(accelerator.device)
                            sample_prompt = next(iter(test_dataloader))['caption']

                            # Extract the images
                            with torch.cuda.amp.autocast():
                                images = val_pipe(
                                    prompt_embeds=val_pipe._encode_prompt(
                                        sample_prompt,
                                        device=accelerator.device,
                                        num_images_per_prompt=1,
                                        do_classifier_free_guidance=True,
                                    ),
                                    image_garm=img_style,
                                    image_vton=img_given,
                                    image_ori=img_given,
                                    num_inference_steps=20,
                                    image_guidance_scale=1.5,
                                    generator=torch.manual_seed(0)
                                ).images

                            # Save the generated images
                            for i, image in enumerate(images):
                                image.save(f"{args.output_dir}/sample_{global_step}_{i}.png")

                            # Compute the metrics
                            ### TODO: Bizim datasetimiz için çalışıyor mu bi bak. tüm metrikleri veren bir fonksiyon
                            """
                            metrics = compute_metrics(
                                os.path.join(args.output_dir, f"imgs_step_{global_step}_{args.test_order}"),
                                args.test_order,
                                args.dataset, 'all', ['all'], args.dresscode_dataroot, args.vitonhd_dataroot)

                            print(metrics, flush=True)
                            accelerator.log(metrics, step=global_step)
                            """    
                            # Delete old checkpoints
                            dirs = os.listdir(os.path.join(args.output_dir, "checkpoint"))
                            dirs = [d for d in dirs if d.startswith("checkpoint")]
                            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                            try:
                                path = dirs[-2]
                                shutil.rmtree(os.path.join(args.output_dir, "checkpoint", path), ignore_errors=True)
                            except:
                                print("No checkpoint to delete")

                            # Save the unet
                            unet_vton_path = os.path.join(args.output_dir, f"unet_vton_{global_step}.pth")
                            unet_garm_path = os.path.join(args.output_dir, f"unet_garm_{global_step}.pth")
                            accelerator.save(unwrapped_unet_vton.state_dict(), unet_vton_path)
                            accelerator.save(unwrapped_unet_garm.state_dict(), unet_garm_path)

                            del unwrapped_unet_garm
                            del unwrapped_unet_vton
                            del val_pipe


                        unet_vton.train()
                        unet_garm.train()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # End of training
    accelerator.wait_for_everyone()
    accelerator.end_training()

def prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
    shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device)

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * self.scheduler.init_noise_sigma
    return latents



def set_seed(seed):
    """
    Set seed for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    accelerate.utils.set_seed(seed)

if __name__ == "__main__":
    main()