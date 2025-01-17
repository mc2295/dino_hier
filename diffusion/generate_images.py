from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
import os
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Simple image generation script.")
    parser.add_argument(
        "--path_to_model_dir",
        type=str,
        default=None,
        required=True,
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
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproduciblity.")

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
        "--num_inference_steps",
        type=int,
        default=100,
        help=(
           "Number of inference steps using in the image generation process"
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help=(
           "Batch size for inference"
        ),
    )

    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=64,
        help=(
           "Number of images to generate per prompt"
        ),
    )

    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts to be generated."),
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

    args = parser.parse_args()


    if args.output_dir is None:
        raise ValueError("Please provide an output dir")

    return args

def main():
    # Load the model
    args = parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = 'cpu'

    print(f"Generating images on device: {device}")

    # If passed along, set the training seed now.
    if args.seed is None:
        args.seed = 42

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    unet = UNet2DConditionModel.from_pretrained(
        args.path_to_model_dir, subfolder="unet"
    )

    # noise_scheduler = DDPMScheduler.from_pretrained(args.path_to_model_dir, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.path_to_model_dir, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
            args.path_to_model_dir, subfolder="text_encoder"
        )
    vae = AutoencoderKL.from_pretrained(
            args.path_to_model_dir, subfolder="vae"
        )


    pipeline = StableDiffusionPipeline.from_pretrained(
        args.path_to_model_dir,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        safety_checker=None,
        torch_dtype=args.mixed_precision,
    )

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    for prompt in args.prompts:
        print(f"Generating images for:\n {prompt}")

        images = []
        for i in range(0,args.num_images_per_prompt,args.batch_size):
            with torch.autocast(device):
                imgs = pipeline(prompt,num_images_per_prompt=args.batch_size,
                 num_inference_steps=args.num_inference_steps,
                 generator=generator).images
            images.extend(imgs)
        str_prompt = prompt.replace(' ','_')
        out_dir_temp = os.join(args.output_dir,str_prompt) 
        os.makedirs(out_dir_temp,exist_ok=True)
        print(f"Saving generated images in {out_dir_temp}")
        for num , image in enumerate(images):
            image.save(os.join(out_dir_temp,f'{num}.jpg'))

    del pipeline
    torch.cuda.empty_cache()



if __name__=='__main__':
    main()