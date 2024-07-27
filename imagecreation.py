import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!

# Load model on CPU
unet = UNet2DConditionModel.from_config(base, subfolder="unet")
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cpu"))


pipe = StableDiffusionXLPipeline.from_pretrained(base,
                                                 unet=unet,
                                                 torch_dtype=torch.float32,  # Ensure consistent dtype
                                                 variant="fp16")

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")


custom_objects = ["animal"]


i=1
a=10    #ilosc

while (i!=0):
    i=i-1
    while (a!=0):
        result = pipe(custom_objects[i-1], num_inference_steps=4, guidance_scale=0, image_size=(32,32)).images[0]
        name = str(f'{custom_objects[i-1]}_{a}.png')

        result = result.convert('L')    #to grayscale - czyli ma 1 kanal
        result32 = result.resize((28,28)) #result.resize((width, height))


        directory = "./datasets/image/"
        directory32 = "./datasets/image32/"
        print (directory+name)
        result.save(directory+name)

        result32.save(directory32+name)
        a=a-1










