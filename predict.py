import os
import time
import sys
import pathlib
from version import MODEL_CACHE, MODEL_ID, REVISION, SAFETY_MODEL_ID, SAFETY_REVISION

check = pathlib.Path("/tmp/predict-import")
if not check.exists():
    check.touch()
else:
    print("===!!!!!!predict has been imported again!!!!!!===")
pget_proc = subprocess.Popen(
    ["/bin/pget", "-x", os.environ["WEIGHT_URL"], MODEL_CACHE], close_fds=True
)


def logtime(msg: str) -> None:
    print(f"===TIME {time.time():.4f} {msg}===", file=sys.stderr)


from typing import List

logtime("importing torch")
import torch

logtime("imported torch, importing cog")
from cog import BasePredictor, Input, Path

logtime("importing cog, importing diffusers")
from diffusers import (
    StableDiffusionPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

logtime("imported diffusers, importing transformers")
from transformers import CLIPFeatureExtractor


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        logtime("predict setup")
        pget_proc.wait()
        logtime("finished pget")
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_MODEL_ID,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            torch_dtype=torch.float16,
            revision=SAFETY_REVISION,
            use_safetensors=True,
        )
        logtime("loaded safety checker")
        # ? wasn't previously necessary
        feature_extractor = CLIPFeatureExtractor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=MODEL_CACHE,
            torch_dtype=torch.float16,
            local_files_only=True,
            use_safetensors=True,
        )
        logtime("loaded feature extractor")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            revision=REVISION,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        logtime("loaded pipe")
        self.pipe = self.pipe.to("cuda")
        logtime("moved pipe to cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a photo of an astronaut riding a horse on mars",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
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
            default="DPMSolverMultistep",
            choices=[
                "DDIM",
                "K_EULER",
                "DPMSolverMultistep",
                "K_EULER_ANCESTRAL",
                "PNDM",
                "KLMS",
            ],
            description="Choose a scheduler.",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        logtime("predict predict start")
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)
        logtime("actually doing prediction")
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
        logtime("got pipe output")

        output_paths = []
        for i, sample in enumerate(output.images):
            if output.nsfw_content_detected and output.nsfw_content_detected[i]:
                continue

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))
        logtime("saved files")

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]
