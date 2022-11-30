import os
from typing import List

import torch
from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionPipeline,
)

from PIL import Image
from cog import BasePredictor, Input, Path

MODEL_ID = "stabilityai/stable-diffusion-2"
MODEL_CACHE = "diffusers-cache"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a photo of an astronaut riding a horse on mars",
        ),
        negative_prompt: str = Input(
            description="The prompt NOT to guide the image generation. Ignored when not using guidance",
            default=None,
        ),
        # width: int = Input(
        #     description="Width of output image. Currently for sd-v2 only allowing 512 and 768 currently.",
        #     choices=[512, 768],
        #     default=512,
        # ),
        # height: int = Input(
        #     description="Height of output image. Currently for sd-v2 only allowing 512 and 768.",
        #     choices=[512, 768],
        #     default=512,
        # ),
        prompt_strength: float = Input(
            description="Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image",
            default=0.8,
        ),
        num_outputs: int = Input(
            description="Number of images to output. Currenly allowing 1-3, otherwise would OOM.",
            ge=1,
            le=3,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="K_EULER",
            choices=["DDIM", "K_EULER"],
            description="Choose a scheduler. Seems only DDIM and K_EULER work for sd-v2 now.",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        width, height = 768, 768

        pipe = self.txt2img_pipe

        pipe.scheduler = make_scheduler(scheduler)

        generator = torch.Generator("cuda").manual_seed(seed)
        output = pipe(
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


def make_scheduler(name):
    return {
        "PNDM": PNDMScheduler.from_pretrained(
            MODEL_ID,
            subfolder="scheduler",
        ),
        "KLMS": LMSDiscreteScheduler.from_pretrained(
            MODEL_ID,
            subfolder="scheduler",
        ),
        "DDIM": DDIMScheduler.from_pretrained(
            MODEL_ID,
            subfolder="scheduler",
        ),
        "K_EULER": EulerDiscreteScheduler.from_pretrained(
            MODEL_ID,
            subfolder="scheduler",
        ),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_pretrained(
            MODEL_ID,
            subfolder="scheduler",
        ),
    }[name]

