import json
import os
from pathlib import Path
import tempfile
import time
import cProfile

import torch
from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    StableDiffusionPipeline,
    UNet2DConditionModel,
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

from transformers import (
    AutoTokenizer,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPConfig,
    CLIPImageProcessor
)

from tensorizer import TensorDeserializer, TensorSerializer, stream_io
from tensorizer.utils import no_init_or_tensor, convert_bytes, get_mem_usage

# MODEL_ID refers to a diffusers-compatible model on HuggingFace
# e.g. prompthero/openjourney-v2, wavymulder/Analog-Diffusion, etc
MODEL_ID = "stabilityai/stable-diffusion-2-1"
MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"

def load_pipe():
    st = time.time()
    print("Loading pipeline...")
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        SAFETY_MODEL_ID,
        cache_dir=MODEL_CACHE,
        local_files_only=True,
        torch_dtype=torch.float16
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        safety_checker=safety_checker,
        cache_dir=MODEL_CACHE,
        local_files_only=True,
        torch_dtype=torch.float16
    ).to("cuda")
    print(f"started in {time.time() - st}")
    return pipe

# testing tensorization
TENSOR_PATH = "tensors"
# tensorize model
def tensorize_model():
    
    pipe = load_pipe()
    for k, component in pipe.components.items():
        if isinstance(component, torch.nn.Module):
            print(f"tensorizing {k}")
            path = os.path.join(TENSOR_PATH, f"{k}.tensors")
            serializer = TensorSerializer(path)
            serializer.write_module(component)
            serializer.close()


def load_model(
    path_uri: str,
    model_class,
    config_class = None,
    model_prefix = "model",
    device = "cuda",
    dtype = None,
) -> torch.nn.Module:
    """
    Given a path prefix, load the model with a custom extension

    Args:
        path_uri: path to the model. Can be a local path or a URI
        model_class: The model class to load the tensors into.
        config_class: The config class to load the model config into. This must be
            set if you are loading a model from HuggingFace Transformers.
        model_prefix: The prefix to use to distinguish between multiple serialized
            models. The default is "model".
        device: The device onto which to load the model.
        dtype: The dtype to load the tensors into. If None, the dtype is inferred from
            the model.
    """

    if model_prefix is None:
        model_prefix = "model"

    begin_load = time.time()
    ram_usage = get_mem_usage()

    config_uri = f"{path_uri}/{model_prefix}_config.json"
    tensors_uri = f"{path_uri}/{model_prefix}.tensors"
    tensor_stream = stream_io.open_stream(tensors_uri)

    print(f"Loading {tensors_uri}, {ram_usage}")

    tensor_deserializer = TensorDeserializer(
        tensor_stream, device=device, dtype=dtype, plaid_mode=True
    )

    if config_class is not None:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_config_path = os.path.join(temp_dir, "config.json")
                with open(temp_config_path, "wb") as temp_config:
                    temp_config.write(stream_io.open_stream(config_uri).read())
                config = config_class.from_pretrained(temp_dir)
                #config.gradient_checkpointing = True
        except ValueError:
            config = config_class.from_pretrained(config_uri)
        with no_init_or_tensor():
            # AutoModels instantiate from a config via their from_config()
            # method, while other classes can usually be instantiated directly.
            config_loader = getattr(model_class, "from_config", model_class)
            model = config_loader(config)
    else:
        try:
            config = json.loads(
                stream_io.open_stream(config_uri).read().decode("utf-8")
            )
        except ValueError:
            with open(config_uri, "r") as f:
                config = json.load(f)
        with no_init_or_tensor():
            model = model_class(**config)

    tensor_deserializer.load_into_module(model)

    tensor_load_s = time.time() - begin_load
    rate_str = convert_bytes(
        tensor_deserializer.total_bytes_read / tensor_load_s
    )
    tensors_sz = convert_bytes(tensor_deserializer.total_bytes_read)
    print(
        f"Model tensors loaded in {tensor_load_s:0.2f}s, read "
        f"{tensors_sz} @ {rate_str}/s, {get_mem_usage()}"
    )

    return model
    
def load_tensorized_models():
    """
    Stolen tensorizer example for loading stable idffusion
    """
    with cProfile.Profile() as pr:
        st = time.time()
        device = "cuda"
        output_prefix = "/src/tensors"
        # ick. 
        cache_dir = '/src/diffusers-cache/models--stabilityai--stable-diffusion-2-1/snapshots/845609e6cf0a060d8cd837297e5c169df5bff72c/'

        # load vae, text encoder, unet, safety_checker
        vae = load_model(output_prefix, AutoencoderKL, None, "vae", device)
        unet = load_model(
            output_prefix, UNet2DConditionModel, None, "unet", device
        )
        encoder = load_model(
            output_prefix, CLIPTextModel, CLIPTextConfig, "text_encoder", device
        )
        safety_checker = load_model(output_prefix, StableDiffusionSafetyChecker, CLIPConfig, "safety_checker", device)

        pipeline = StableDiffusionPipeline(
            text_encoder=encoder,
            vae=vae,
            unet=unet,
            tokenizer=AutoTokenizer.from_pretrained(
                cache_dir, subfolder="tokenizer"
            ),
            scheduler=LMSDiscreteScheduler.from_pretrained(
                cache_dir, subfolder="scheduler"
            ),
            safety_checker=safety_checker,
            feature_extractor=CLIPImageProcessor.from_pretrained(
                os.path.join(cache_dir, "feature_extractor")
            )
        ).to(device)
        print(f"Took {time.time() - st} to load tensorized models")
        pr.dump_stats(f"{st:.0f}_setup.prof")

    return pipeline

def test_stable_diffusion_generation(prompt="a big yellow dog, trending on artstation", width=512, height=512, guidance_scale=7.5, num_inference_steps=50, num_outputs=1, negative_prompt=None):
    pipe = load_tensorized_models()

    seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")

    pipe.scheduler = make_scheduler("DDIM", pipe.scheduler.config)

    generator = torch.Generator("cuda").manual_seed(seed)
    output = pipe(
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

    output_paths = []
    for i, sample in enumerate(output.images):
        if output.nsfw_content_detected and output.nsfw_content_detected[i]:
            continue

        output_path = f"/src/out-{i}.png"
        sample.save(output_path)
        output_paths.append(Path(output_path))

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
