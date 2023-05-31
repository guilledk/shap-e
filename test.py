import os
import gc
import torch

from pathlib import Path
import subprocess

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images
from shap_e.util.collections import AttrDict
from shap_e.models.transmitter.base import Transmitter

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # full determinism
    # https://huggingface.co/docs/diffusers/using-diffusers/reproducibility#deterministic-algorithms
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

batch_size = 1
guidance_scale = 10.0
prompt = "starship"
size = 128 # this is the size of the renders; higher values take longer to render.

latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(texts=[prompt] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
)

gc.collect()
torch.cuda.empty_cache()

render_mode = 'nerf' # you can change this to 'stf'

cameras = create_pan_cameras(size, device)
images = decode_latent_images(xm, latents[0], cameras, rendering_mode=render_mode)

for i, image in enumerate(images):
    Path(f'gifs/{prompt}').mkdir(exist_ok=True)
    image.save(f'gifs/{prompt}/frame{i}.png')

out = subprocess.run([
    'ffmpeg', '-i', f'gifs/{prompt}/frame%01d.png',
    '-c:v', 'libx264', '-vf', 'fps=12', '-pix_fmt', 'yuv420p',
    f'gifs/{prompt}/out.mp4'])

print(out)
