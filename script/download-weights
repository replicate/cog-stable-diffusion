#!/usr/bin/env python

import os
import sys

import torch
from diffusers import StableDiffusionPipeline

os.makedirs("diffusers-cache", exist_ok=True)


pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    cache_dir="diffusers-cache",
    use_auth_token=sys.argv[1],
)
