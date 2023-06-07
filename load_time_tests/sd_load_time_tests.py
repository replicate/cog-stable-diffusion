import argparse
import json
import os
import time
import timeit

import torch
from diffusers import (AutoencoderKL, LMSDiscreteScheduler,
                       StableDiffusionPipeline, UNet2DConditionModel)
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from tensorizer import TensorDeserializer, stream_io
from tensorizer.utils import convert_bytes, get_mem_usage, no_init_or_tensor
from transformers import (AutoTokenizer, CLIPConfig, CLIPImageProcessor,
                          CLIPTextConfig, CLIPTextModel)


def load_with_from_pretrained(path, use_safetensors=False, device='cpu', load_safety_checker=False):
    """
    Loads Stable Diffusion pipeline, optionally including safetensors
    """
    st = time.time()
    safety_checker = None
    if load_safety_checker:
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            os.path.join(path, 'safety_checker'), 
            torch_dtype=torch.float16,
            local_files_only=True
        )
    
    txt2img_pipe = StableDiffusionPipeline.from_pretrained(
        path,
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=safety_checker
    )
    print(f"load time: {time.time() - st}")
    if device=="cuda":
        txt2img_pipe.to("cuda")
        print(f"total time: {time.time() - st}")
    return txt2img_pipe


def _load_single_tensorized_model(
    path_uri: str,
    model_class,
    config_class = None,
    model_prefix = "model",
    device = "cuda",
    dtype = None,
) -> torch.nn.Module:
    """
    Lifted from tensorizer examples - basically a general purpose diffusers/transformers module instantiator w/tensorizer

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

    config_uri = f"{path_uri}/{model_prefix}/config.json"
    tensors_uri = f"{path_uri}/{model_prefix}/{model_prefix}.tensors"
    tensor_stream = stream_io.open_stream(tensors_uri)

   #print(f"Loading {tensors_uri}, {ram_usage}")

    plaid_mode = True if device == 'cuda' else False
    tensor_deserializer = TensorDeserializer(
        tensor_stream, device=device, dtype=dtype, plaid_mode=plaid_mode
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

    # tensor_load_s = time.time() - begin_load
    # rate_str = convert_bytes(
    #     tensor_deserializer.total_bytes_read / tensor_load_s
    # )
    # tensors_sz = convert_bytes(tensor_deserializer.total_bytes_read)
    # print(
    #     f"Model tensors loaded in {tensor_load_s:0.2f}s, read "
    #     f"{tensors_sz} @ {rate_str}/s, {get_mem_usage()}"
    # )

    return model
    
def load_tensorized_models(path="/weights/fp16/tensors", load_safety_checker=False, device="cpu"):
    """
    Stolen tensorizer example for loading stable idffusion
    """
    st = time.time()

    # load all torch models
    vae = _load_single_tensorized_model(path, AutoencoderKL, None, "vae", device)
    unet = _load_single_tensorized_model(
        path, UNet2DConditionModel, None, "unet", device
    )
    encoder = _load_single_tensorized_model(
        path, CLIPTextModel, CLIPTextConfig, "text_encoder", device
    )
    safety_checker = None
    if load_safety_checker:
        safety_checker = _load_single_tensorized_model(path, StableDiffusionSafetyChecker, CLIPConfig, "safety_checker", device)

    pipeline = StableDiffusionPipeline(
        text_encoder=encoder,
        vae=vae,
        unet=unet,
        tokenizer=AutoTokenizer.from_pretrained(
            path, subfolder="tokenizer"
        ),
        scheduler=LMSDiscreteScheduler.from_pretrained(
            path, subfolder="scheduler"
        ),
        safety_checker=safety_checker,
        feature_extractor=CLIPImageProcessor.from_pretrained(
            os.path.join(path, "feature_extractor")
        )
    )

    if device == "cuda":
        pipeline = pipeline.to(device)
    print(f"Took {time.time() - st} to load tensorized models")

    return pipeline



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--from-pretrained', action='store_true')
    parser.add_argument('--safetensors', action='store_true')
    parser.add_argument('--tensorizer', action='store_true')
    parser.add_argument('--accelerate', action='store_true')
    parser.add_argument('--safety-checker', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if args.cuda else 'cpu'
    kwargs = dict(load_safety_checker=args.safety_checker, device=device)

    # Test load times to CPU
    if args.from_pretrained and args.safetensors:
        load_fn = load_with_from_pretrained
        kwargs['path'] = '/src/weights/fp16/safetensors/'
        kwargs['use_safetensors'] = args.safetensors

        #print(f"Loading from {kwargs['path']} with load_fn={load_fn} and use_safetensors={args.safetensors}")
        print(f"Loading with {kwargs}")
        elpsd = timeit.timeit(lambda: load_fn(**kwargs), number=1)

    elif args.from_pretrained and not args.safetensors:
        load_fn = load_with_from_pretrained
        kwargs['path'] = '/src/weights/fp16/bin/'
        kwargs['use_safetensors'] = args.safetensors

        #print(f"Loading from {kwargs['path']} with load_fn={load_fn} and use_safetensors={args.safetensors}")
        print(f"Loading with {kwargs}")
        elpsd = timeit.timeit(lambda: load_fn(**kwargs), number=1)
    
    elif args.tensorizer:
        load_fn = load_tensorized_models
        kwargs['path'] = '/src/weights/fp16/tensors'
        #print(f"Loading from {kwargs['path']} with load_fn={load_fn}")
        print(f"Loading with {kwargs}")
        elpsd = timeit.timeit(lambda: load_fn(**kwargs), number=1)

    elif args.accelerate and args.safetensors:
        load_fn = load_with_accelerate
        kwargs['path'] = 'weights/fp16/safetensors/unet'
        kwargs['use_safetensors'] = args.use_safetensors

        print(f"Loading from {kwargs['path']} with load_fn={load_fn} and use_safetensors={args.safetensors}")
        elpsd = timeit.timeit(lambda: load_fn(**kwargs), number=1)
    
    print("Elapsed time: ", elpsd, " seconds")