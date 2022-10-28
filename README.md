# Stable Diffusion Cog model

[![Replicate](https://replicate.com/stability-ai/stable-diffusion/badge)](https://replicate.com/stability-ai/stable-diffusion)

This is an implementation of the [Diffusers Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights [with your Hugging Face auth token](https://huggingface.co/settings/tokens):

    cog run script/download-weights <your-hugging-face-auth-token>

Then, you can run predictions:

    cog predict -i prompt="monkey scuba diving"
