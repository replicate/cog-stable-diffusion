Before `cog predict`, `cog push`, etc., add the pretrained weights to `diffusers-cache`:

```
$ mkdir diffusers-cache
$ python
>>> pipe = StableDiffusionPipeline.from_pretrained(
             "CompVis/stable-diffusion-v1-4",
             cache_dir="diffusers-cache",
             use_auth_token=<your-huggingface-auth-token>,
)
```
