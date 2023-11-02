from PIL import Image
import torch

from cog import BasePredictor, Input, Path as CogPath
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor

MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"


class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_MODEL_ID, cache_dir=MODEL_CACHE, local_files_only=True
        ).to(self.device)
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

    def predict(self, image: CogPath = Input(description="Input image")) -> str:
        image = Image.open(image).convert("RGB").resize((512, 512))
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(self.device)
        _, has_nsfw_concepts = self.safety_checker(
            images=[image], clip_input=safety_checker_input.pixel_values
        )
        is_nsfw = any(has_nsfw_concepts)
        print(f"{is_nsfw=}, The image is safe." if not is_nsfw else "The image is not safe.")
        return str(is_nsfw).lower()
