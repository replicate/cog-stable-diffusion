import argparse
import json
import os
import time
import timeit

import diffusers
import torch
import transformers
from diffusers import (AutoencoderKL, LMSDiscreteScheduler,
                       StableDiffusionPipeline, UNet2DConditionModel)
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from tensorizer import TensorDeserializer, stream_io
from tensorizer.utils import convert_bytes, get_mem_usage, no_init_or_tensor
from transformers import (AutoTokenizer, CLIPConfig, CLIPImageProcessor,
                          CLIPTextConfig, CLIPTextModel)


def load_with_from_pretrained(path, use_safetensors=False):
    """
    Loads Stable Diffusion pipeline, optionally including safetensors
    """
    st = time.time()
    txt2img_pipe = StableDiffusionPipeline.from_pretrained(
        path,
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
    )
    print(f"load time: {time.time() - st}")
    txt2img_pipe.to("cuda")
    print(f"total time: {time.time() - st}")
    return txt2img_pipe

def load_with_tensorizer(component_map):
    """
    Loads stable diffusion pipeline with tensorizer
    """
    st = time.time()
    components = {"scheduler": diffusers.schedulers.scheduling_ddim.DDIMScheduler, "safety_checker": False}
    for k in component_map.keys():
        print(f'Loading {k}...')
        cls = component_map[k].get('cls')
        path = component_map[k].get('path')
        tensorized_weights = component_map[k].get('tensorized_weights', None)

        if tensorized_weights:
            with no_init_or_tensor():
                model = cls.from_pretrained(path)
            
            deserializer = TensorDeserializer(tensorized_weights, plaid_mode=True)
            deserializer.load_into_module(model)

            components[k] = model
        
        else:
            model = cls.from_pretrained(path)
            components[k] = model
                
    pipe = diffusers.StableDiffusionPipeline(**components)
    print(f"Load time: {time.time() - st}")
    pipe = pipe.to('cuda')
    print(f"Total time: {time.time() - st}")
    return pipe


def load_model(
    path_uri: str,
    model_class,
    config_class = None,
    model_prefix = "model",
    device = "cuda",
    dtype = None,
) -> torch.nn.Module:
    """
    Given a path prefix, load the tensorized model with a custom extension. 
    Lifted from tensorizer examples - basically a general purpose diffusers/transformers module instantiator

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
        # try:
        #     with tempfile.TemporaryDirectory() as temp_dir:
        #         temp_config_path = os.path.join(temp_dir, "config.json")
        #         with open(temp_config_path, "wb") as temp_config:
        #             temp_config.write(stream_io.open_stream(config_uri).read())
        #         config = config_class.from_pretrained(temp_dir)
        #         #config.gradient_checkpointing = True
        # except ValueError:
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
    st = time.time()
    device = "cuda"
    tensor_dir = "/weights/fp16/tensors"

    # load all torch models
    vae = load_model(tensor_dir, AutoencoderKL, None, "vae", device)
    unet = load_model(
        tensor_dir, UNet2DConditionModel, None, "unet", device
    )
    encoder = load_model(
        tensor_dir, CLIPTextModel, CLIPTextConfig, "text_encoder", device
    )
    safety_checker = load_model(tensor_dir, StableDiffusionSafetyChecker, CLIPConfig, "safety_checker", device)

    pipeline = StableDiffusionPipeline(
        text_encoder=encoder,
        vae=vae,
        unet=unet,
        tokenizer=AutoTokenizer.from_pretrained(
            tensor_dir, subfolder="tokenizer"
        ),
        scheduler=LMSDiscreteScheduler.from_pretrained(
            tensor_dir, subfolder="scheduler"
        ),
        safety_checker=safety_checker,
        feature_extractor=CLIPImageProcessor.from_pretrained(
            os.path.join(tensor_dir, "feature_extractor")
        )
    ).to(device)
    print(f"Took {time.time() - st} to load tensorized models")

    return pipeline





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--from-pretrained', action='store_true')
    parser.add_argument('--safetensors', action='store_true')
    parser.add_argument('--tensorizer', action='store_true')
    parser.add_argument('--tensorizer_2', action='store_true')
    parser.add_argument('--accelerate', action='store_true')
    args = parser.parse_args()

    # Test load times to CPU
    if args.from_pretrained and args.safetensors:
        load_fn = load_with_from_pretrained
        path = 'weights/fp16/safetensors/'
        kwargs = dict(path=path, use_safetensors=args.safetensors)

        print(f"Loading from {path} with load_fn={load_fn} and use_safetensors={args.safetensors}")
        elpsd = timeit.timeit(lambda: load_fn(**kwargs), number=1)

    elif args.from_pretrained and not args.safetensors:
        load_fn = load_with_from_pretrained
        path = 'weights/fp16/bin/'
        kwargs = dict(path=path, use_safetensors=args.safetensors)

        print(f"Loading from {path} with load_fn={load_fn} and use_safetensors={args.safetensors}")
        elpsd = timeit.timeit(lambda: load_fn(**kwargs), number=1)
    
    elif args.tensorizer_2:
        load_fn = load_tensorized_models
        path = 'weights/fp16/tensors'
        print(f"Loading from {path} with load_fn={load_fn}")
        elpsd = timeit.timeit(lambda: load_fn(**kwargs), number=1)


    elif args.tensorizer:
        from tensorizer import TensorDeserializer
        from tensorizer.utils import no_init_or_tensor

        base_path = 'weights/fp16/tensors/'
        component_map = {
            'vae': {
                    'tensorized_weights': os.path.join(base_path, 'vae', 'vae.tensors'),
                    'path': os.path.join(base_path, 'vae'),
                    'cls': diffusers.AutoencoderKL
                },
            'text_encoder': {
                    'tensorized_weights': os.path.join(base_path, 'text_encoder', 'text_encoder.tensors'),
                    'path': os.path.join(base_path, 'text_encoder'),
                    'cls': transformers.models.clip.modeling_clip.CLIPTextModel,
                },
            'unet': {
                    'tensorized_weights': os.path.join(base_path, 'unet', 'unet.tensors'),
                    'path': os.path.join(base_path, 'unet'),
                    'cls': diffusers.models.unet_2d_condition.UNet2DConditionModel
                },
            'safety_checker': {
                    'tensorized_weights': os.path.join(base_path, 'safety_checker', 'safety_checker.tensors'),
                    'path': os.path.join(base_path, 'safety_checker'),
                    'cls': diffusers.pipelines.stable_diffusion.safety_checker.StableDiffusionSafetyChecker
                },
            'tokenizer': {
                    'path': os.path.join(base_path, 'tokenizer'),
                    'cls': transformers.models.clip.tokenization_clip.CLIPTokenizer,
                },
            'feature_extractor': {
                    'path': os.path.join(base_path, 'feature_extractor'),
                    'cls': transformers.models.clip.image_processing_clip.CLIPImageProcessor,
                },
        }

        load_fn = load_with_tensorizer
        kwargs = dict(component_map=component_map)

        print(f"Loading from {base_path} with load_fn={load_fn}")
        elpsd = timeit.timeit(lambda: load_fn(**kwargs), number=1)


    elif args.accelerate and args.safetensors:
        load_fn = load_with_accelerate
        path = 'weights/fp16/safetensors/unet'
        kwargs = dict(path=path, use_safetensors=args.safetensors)

        print(f"Loading from {path} with load_fn={load_fn} and use_safetensors={args.safetensors}")
        elpsd = timeit.timeit(lambda: load_fn(**kwargs), number=1)
    
    print("Elapsed time: ", elpsd, " seconds")