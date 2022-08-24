import os
from typing import List

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from diffusers import PNDMScheduler
from PIL import Image

from image_to_image import StableDiffusionImg2ImgPipeline

MODEL_CACHE = "diffusers-cache"

def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.log_dir = Path("/tmp/cog-stable-diffusion")
        self.log_dir.mkdir(exist_ok=True)

        scheduler = PNDMScheduler()
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            cache_dir=MODEL_CACHE,
            scheduler=scheduler,
            revision="fp16",
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(description="Input prompt", default=""),
        init_image: Path = Input(
            description="Inital image to generate variations of.", default=None
        ),
        strength: float = Input(
            description="Strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image.",
            default=0.8,
            ge=0.0,
            le=1.0,
        ),
        width: int = Input(
            description="Width of output image",
            default=512,
            choices=[256, 384, 512, 640, 768, 896, 1024],
        ),
        height: int = Input(
            description="Height of output image",
            default=512,
            choices=[256, 384, 512, 640, 768, 896, 1024],
        ),
        num_outputs: int = Input(
            description="Number of images to output", choices=[1, 4, 16], default=1
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
        generator = torch.Generator("cuda").manual_seed(seed)
        print(f"Using seed: {seed}")

        if init_image is not None:
            init_image = Image.open(init_image).convert("RGB")
            init_image = preprocess(init_image)
            width, height = init_image.shape[2], init_image.shape[3]
        else:
            init_image = torch.randn(
                1,
                3,
                height,
                width,
                dtype=torch.float16,
                generator=generator,
                device="cuda",
            )
            strength = 1.0  # disable noising

        print(f"Using resolution of {width}x{height} for generation dimensions.")
        prompt = [prompt] * num_outputs

        output = self.pipe(
            prompt=prompt,
            init_image=init_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        )

        output_paths = []
        for i, sample in enumerate(output["sample"]):
            output_path = self.log_dir / f"out-{i:03d}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
