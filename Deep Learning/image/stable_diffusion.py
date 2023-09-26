from diffusers import DiffusionPipeline
import torch


model = "stabilityai/stable-diffusion-xl-base-1.0"
# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 200
high_noise_frac = 0.8
prompt = "A unicorn in an epic battle against a narwhal"

device = "cuda" if torch.cuda.is_available() else "cpu"

# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to(device)
base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
refiner = DiffusionPipeline.from_pretrained(
    model,
    text_encoder=base.text_encoder,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# run both experts
image = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images

image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]

image.save("diffusion.png")

# model = "stabilityai/stable-diffusion-2-1"
