# Stable Diffusion v2 Cog model

[![Replicate](https://replicate.com/cjwbw/stable-diffusion-v2/badge)](https://replicate.com/cjwbw/stable-diffusion-v2) 

This is an implementation of the [Diffusers Stable Diffusion v2](https://huggingface.co/stabilityai/stable-diffusion-2) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)


    cog run script/download-weights 

Then, you can run predictions:

    cog predict -i prompt="monkey scuba diving"
