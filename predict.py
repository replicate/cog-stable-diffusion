import json
import os
from typing import List

import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
)

# Stable Diffusion Defaults
MODEL_ID = "stabilityai/stable-diffusion-2-1"
MODEL_CACHE = "diffusers-cache"
DEFAULT_HEIGHT = 768
DEFAULT_WIDTH = 768
DEFAULT_SCHEDULER = "K_EULER"
DEFAULT_PROMPT = "a photo of an astronaut riding a horse on mars"

# Dreambooth Training Defaults
USE_DREAMBOOTH_WEIGHTS = os.path.exists("weights")
if USE_DREAMBOOTH_WEIGHTS:
    DEFAULT_HEIGHT = 512
    DEFAULT_WIDTH = 512
    DEFAULT_SCHEDULER = "DDIM"
    try:
        with open("weights/args.json") as f:
            args = json.load(f)
            if args.get('instance_prompt'):
                DEFAULT_PROMPT = args['instance_prompt']
    except:
        pass


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        if USE_DREAMBOOTH_WEIGHTS:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "weights",
                safety_checker=None,
                torch_dtype=torch.float16,
            ).to("cuda")
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                MODEL_ID,
                cache_dir=MODEL_CACHE,
                local_files_only=True,
            ).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default=DEFAULT_PROMPT,
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=DEFAULT_WIDTH,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=DEFAULT_HEIGHT,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image",
            default=0.8,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default=DEFAULT_SCHEDULER,
            choices=[
                "DDIM",
                "K_EULER",
                "DPMSolverMultistep",
            ],
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

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            negative_prompt=[negative_prompt] * num_outputs
            if negative_prompt is not None
            else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths


def make_scheduler(name, config):
    return {
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]
