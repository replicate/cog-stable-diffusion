import os
from typing import List

import torch
from cog import BasePredictor, Input, Path
from diffusers import PNDMScheduler
from PIL import Image

from image_to_image import StableDiffusionImg2ImgPipeline, preprocess


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.model_cache = "model_cache"

        scheduler = PNDMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            cache_dir="diffusers-cache",
            scheduler=scheduler,
            revision="fp16",
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
        init_image: Path = Input(
            description="Inital image to generate variations of.", default=None
        ),
        strength: float = Input(
            description="Strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image.",
            default=0.8,
        ),
        num_outputs: int = Input(
            description="Number of images to output", choices=[1, 4, 16], default=4
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=100
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        init_image = Image.open(init_image).convert("RGB")
        init_image = init_image.resize((512, 512))
        init_image = preprocess(init_image)
        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            init_image=init_image,
            prompt=[prompt] * num_outputs if prompt is not None else None,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            strength=strength,
        )
        if any(output["nsfw_content_detected"]):
            raise Exception("NSFW content detected, please try a different prompt")

        output_paths = []
        for i, sample in enumerate(output["sample"]):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
