

def load_with_from_pretrained(path: str, use_safetensors=False):
    unet = UNet2DConditionModel.from_pretrained(
        path,
        torch_dtype=torch.float16,
        use_safetensors=use_safetensors
    )

def load_with_accelerate(path: str, use_safetensors=False):


    config = UNet2DConditionModel.load_config(path)

    with init_empty_weights():
        model = UNet2DConditionModel.from_config(config, torch_dtype=torch.float16)
    
    model = load_checkpoint_and_dispatch(
        model, checkpoint = os.path.join(path, "diffusion_pytorch_model.safetensors"), device_map={"": 0}
    )

if __name__ == "__main__":
    import argparse
    import timeit
    import os
    import torch
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from diffusers import (
        UNet2DConditionModel
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-pretrained', action='store_true')
    parser.add_argument('--safetensors', action='store_true')
    parser.add_argument('--accelerate', action='store_true')
    args = parser.parse_args()

    if args.from_pretrained and args.safetensors:
        load_fn = load_with_from_pretrained
        path = 'weights/fp16/safetensors/unet'
        kwargs = dict(path=path, use_safetensors=args.safetensors)

        print(f"Loading from {path} with load_fn={load_fn} and use_safetensors={args.safetensors}")
        elpsd = timeit.timeit(lambda: load_fn(**kwargs), number=1)

    elif args.from_pretrained and not args.safetensors:
        load_fn = load_with_from_pretrained
        path = 'weights/fp16/bin/unet'
        kwargs = dict(path=path, use_safetensors=args.safetensors)

        print(f"Loading from {path} with load_fn={load_fn} and use_safetensors={args.safetensors}")
        elpsd = timeit.timeit(lambda: load_fn(**kwargs), number=1)

    elif args.accelerate and args.safetensors:
        load_fn = load_with_accelerate
        path = 'weights/fp16/safetensors/unet'
        kwargs = dict(path=path, use_safetensors=args.safetensors)

        print(f"Loading from {path} with load_fn={load_fn} and use_safetensors={args.safetensors}")
        elpsd = timeit.timeit(lambda: load_fn(**kwargs), number=1)
    
    print("Elapsed time: ", elpsd, " seconds")
