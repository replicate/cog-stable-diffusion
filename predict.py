from torch import autocast
import os
import torch
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        lms = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            scheduler=lms,
            cache_dir="diffusers-cache",
            local_files_only=True,
        ).to("cuda")

    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=100
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        seed: int = Input(description="Random seed. Random if less than 0", default=-1),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed < 0:
            seed = int.from_bytes(os.urandom(2), "big")

        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            prompt,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        if output["nsfw_content_detected"][0]:
            raise Exception("NSFW content detected, please try a different prompt")

        output_path = "/tmp/out.png"
        output["sample"][0].save(output_path)

        return Path(output_path)
