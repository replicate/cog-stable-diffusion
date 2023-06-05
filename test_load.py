import os
import time
from typing import List

import torch
from cog import BasePredictor, Input, Path
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

# MODEL_ID refers to a diffusers-compatible model on HuggingFace
# e.g. prompthero/openjourney-v2, wavymulder/Analog-Diffusion, etc
MODEL_ID = "stabilityai/stable-diffusion-2-1"
MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        st = time.time()
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "/src/weights/safety_checker",
            torch_dtype = torch.float16
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "/src/weights",
            safety_checker=safety_checker,
            torch_dtype = torch.float16
        )
        print(f'loaded in {time.time() - st}')

    def predict(self) -> str:
        return "hi"
    
if __name__ == '__main__':
    p = Predictor()
    p.setup()
    
