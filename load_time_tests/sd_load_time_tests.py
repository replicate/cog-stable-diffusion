import time

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




if __name__ == "__main__":
    import argparse
    import timeit
    import os
    from diffusers import StableDiffusionPipeline
    import torch
    import diffusers
    import transformers
    


    parser = argparse.ArgumentParser()
    parser.add_argument('--from-pretrained', action='store_true')
    parser.add_argument('--safetensors', action='store_true')
    parser.add_argument('--tensorizer', action='store_true')
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
            # 'safety_checker': {
            #         'tensorized_weights': os.path.join(base_path, 'safety_checker', 'safety_checker.tensors'),
            #         'path': os.path.join(base_path, 'safety_checker'),
            #         'cls': diffusers.pipelines.stable_diffusion.safety_checker.StableDiffusionSafetyChecker
            #     },
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