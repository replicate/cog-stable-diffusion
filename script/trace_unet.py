import time
import torch
from diffusers import StableDiffusionPipeline
import functools

MODEL_CACHE = "diffusers-cache"

# torch disable grad
torch.set_grad_enabled(False)

# set variables
n_experiments = 2
unet_runs_per_experiment = 50

# load inputs
def generate_inputs():
    sample = torch.randn(2, 4, 64, 64).cuda()
    timestep = torch.rand(1).cuda() * 999
    encoder_hidden_states = torch.randn(2, 77, 1024).cuda()
    return sample, timestep, encoder_hidden_states


pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    cache_dir=MODEL_CACHE
).to("cuda")
unet = pipe.unet
unet.eval()
#unet.to(memory_format=torch.channels_last)  nah
unet.forward = functools.partial(unet.forward, return_dict=False)  # set return_dict=False as default

# warmup
for _ in range(3):
    with torch.inference_mode():
        inputs = generate_inputs()
        orig_output = unet(*inputs)

# trace
print("tracing..")
unet_traced = torch.jit.trace(unet, inputs)
unet_traced.eval()
print("done tracing")


# warmup and optimize graph
for _ in range(5):
    with torch.inference_mode():
        inputs = generate_inputs()
        orig_output = unet_traced(*inputs)


# benchmarking
with torch.inference_mode():
    for _ in range(n_experiments):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(unet_runs_per_experiment):
            orig_output = unet_traced(*inputs)
        torch.cuda.synchronize()
        print(f"unet traced inference took {time.time() - start_time:.2f} seconds")
    for _ in range(n_experiments):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(unet_runs_per_experiment):
            orig_output = unet(*inputs)
        torch.cuda.synchronize()
        print(f"unet inference took {time.time() - start_time:.2f} seconds")

# save the model
unet_traced.save("unet_traced.pt")