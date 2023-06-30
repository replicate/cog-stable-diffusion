still kind of a manual process, install py3.11 on a machine, poetry lock, wormhole requirements.txt and poetry.lock back, run depedency_splitter.py


# Stable Diffusion v2 Cog model

[![Replicate](https://replicate.com/stability-ai/stable-diffusion/badge)](https://replicate.com/stability-ai/stable-diffusion) 

This is an implementation of the [Diffusers Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights 

Then, you can run predictions:

    cog predict -i prompt="monkey scuba diving"
