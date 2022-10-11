import os
from typing import List

import torch
from diffusers import PNDMScheduler, LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler
from PIL import Image
from cog import BasePredictor, Input, Path

from image_to_image import (
    StableDiffusionImg2ImgPipeline,
    preprocess_init_image,
    preprocess_mask,
)


MODEL_CACHE = "diffusers-cache"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(description="Input prompt", default=""),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 512, 768, 896, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 512, 768, 896, 1024],
            default=512,
        ),
        init_image: Path = Input(
            description="Inital image to generate variations of. Will be resized to the specified width and height",
            default=None,
        ),
        mask: Path = Input(
            description="Black and white image to use as mask for inpainting over init_image. Black pixels are inpainted and white pixels are preserved. Experimental feature, tends to work better with prompt strength of 0.5-0.7",
            default=None,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image",
            default=0.8,
        ),
        num_outputs: int = Input(
            description="Number of images to output. NSFW filter in enabled, so you may get fewer outputs than requested if flagged",
            choices=[1, 4],
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="PNDM",
            choices=["DDIM", "K-LMS", "PNDM", "DDPM"],
            description="Choose a scheduler",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width == height == 1024:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        if init_image:
            init_image = Image.open(init_image).convert("RGB")
            init_image = preprocess_init_image(init_image, width, height).to("cuda")

        self.pipe.scheduler = make_scheduler(scheduler)

        if mask:
            mask = Image.open(mask).convert("RGB")
            mask = preprocess_mask(mask, width, height).to("cuda")

        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            init_image=init_image,
            mask=mask,
            width=width,
            height=height,
            prompt_strength=prompt_strength,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        samples = [
            output["sample"][i]
            for i, nsfw_flag in enumerate(output["nsfw_content_detected"])
            if not nsfw_flag
        ]

        if len(samples) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        print(
            f"NSFW content detected in {num_outputs - len(samples)} outputs, showing the rest {len(samples)} images..."
        )
        output_paths = []
        for i, sample in enumerate(samples):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths


def make_scheduler(name):
    return {
        "PNDM": PNDMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        ),
        "K-LMS": LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        ),
        "DDIM": DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        ),
        "DDPM": DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        ),
    }[name]
