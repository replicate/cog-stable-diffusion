# MODEL_ID refers to a diffusers-compatible model on HuggingFace
# e.g. prompthero/openjourney-v2, wavymulder/Analog-Diffusion, etc
MODEL_ID = "stabilityai/stable-diffusion-2-1"
MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"
# these are both fp16 safetensors I made
REVISION = "refs/pr/71"
SAFETY_REVISION = "refs/pr/41"
