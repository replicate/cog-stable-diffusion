# Stable Diffusion v2 Cog model

[![Replicate](https://replicate.com/stability-ai/stable-diffusion/badge)](https://replicate.com/stability-ai/stable-diffusion) 

This is an implementation of the [Diffusers Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

Make single prediction:
```bash
cog predict -i prompt="monkey scuba diving"
```

Run HTTP API for making predictions:
```bash
cog run -p 5000 python -m cog.server.http
```