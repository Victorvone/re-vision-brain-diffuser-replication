"""
Smoke test: confirms the vendored (frozen, 2023-era fork of SHI-Labs) Versatile
Diffusion code + downloaded checkpoints still load and run end-to-end under the
current environment (Python 3.12, PyTorch 2.12+ROCm, transformers 5.x), using
synthetic CLIP-Text/CLIP-Vision "predicted" features instead of real regression
output. Adapted from scripts/versatilediffusion_reconstruct_images.py, with all
device placement consolidated onto a single GPU (cuda(0)) since this machine has
one GPU, not the two 12GB GPUs the original code assumes.
"""
import sys
sys.path.append('versatile_diffusion')
import os
import PIL
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from lib.model_zoo.vd import VD

print('Libs imported OK')


def regularize_image(x):
    BICUBIC = PIL.Image.Resampling.BICUBIC
    if isinstance(x, str):
        x = Image.open(x).resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, PIL.Image.Image):
        x = x.resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, np.ndarray):
        x = PIL.Image.fromarray(x).resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, torch.Tensor):
        pass
    else:
        assert False, 'Unknown image type'
    assert (x.shape[1] == 512) & (x.shape[2] == 512), 'Wrong image size'
    return x


DEVICE = 'cuda:0'

cfgm_name = 'vd_noema'
sampler_cls = DDIMSampler_VD
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
print('Model instantiated from config OK (this triggers the CLIP download)')

print('Loading main VD checkpoint (critical test: old .pth checkpoint under new torch)...')
sd = torch.load(pth, map_location='cpu')
missing, unexpected = net.load_state_dict(sd, strict=False)
print(f'load_state_dict OK — missing={len(missing)} unexpected={len(unexpected)}')

# Single-GPU consolidation: everything on cuda:0 instead of the original cuda(0)/cuda(1) split
net.clip.to(DEVICE)
net.autokl.to(DEVICE)
net.model.to(DEVICE)
sampler = sampler_cls(net)
batch_size = 1

n_test = 2
rng = np.random.default_rng(0)
# CLIP-Text: (n, 77, 768); CLIP-Vision: (n, 257, 768) — matches the real regression output shapes
pred_text = torch.tensor(rng.standard_normal((n_test, 77, 768)).astype(np.float32)).half().to(DEVICE)
pred_vision = torch.tensor(rng.standard_normal((n_test, 257, 768)).astype(np.float32)).half().to(DEVICE)

n_samples = 1
ddim_steps = 50
ddim_eta = 0
scale = 7.5
strength = 0.75
mixing = 0.4
net.autokl.half()

os.makedirs('smoke_test_output', exist_ok=True)

# Use a real VDVAE-decoded image as the stage-1 input if the VDVAE smoke test already ran,
# otherwise fall back to a synthetic image (this smoke test only checks that the VD code runs).
stage1_path = 'smoke_test_output/vdvae_smoke_0.png'
if os.path.exists(stage1_path):
    zim = Image.open(stage1_path)
    print(f'Using real VDVAE smoke-test output as stage-1 input: {stage1_path}')
else:
    zim = Image.fromarray(rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8))
    print('VDVAE smoke-test output not found, using synthetic noise image as stage-1 input')

torch.manual_seed(0)
for im_id in range(n_test):
    zin = regularize_image(zim)
    zin = zin * 2 - 1
    zin = zin.unsqueeze(0).to(DEVICE).half()

    init_latent = net.autokl_encode(zin)
    print(f'[{im_id}] autokl_encode OK, latent shape {init_latent.shape}')

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    assert 0. <= strength <= 1.
    t_enc = int(strength * ddim_steps)
    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(DEVICE))
    print(f'[{im_id}] stochastic_encode OK')

    dummy = ''
    utx = net.clip_encode_text(dummy)
    utx = utx.to(DEVICE).half()

    dummy_img = torch.zeros((1, 3, 224, 224)).to(DEVICE)
    uim = net.clip_encode_vision(dummy_img)
    uim = uim.to(DEVICE).half()
    print(f'[{im_id}] unconditional CLIP text/vision encode OK')

    z_enc = z_enc.to(DEVICE)
    h, w = 512, 512
    shape = [n_samples, 4, h // 8, w // 8]

    cim = pred_vision[im_id].unsqueeze(0)
    ctx = pred_text[im_id].unsqueeze(0)

    sampler.model.model.diffusion_model.device = DEVICE
    sampler.model.model.diffusion_model.half().to(DEVICE)

    z = sampler.decode_dc(
        x_latent=z_enc,
        first_conditioning=[uim, cim],
        second_conditioning=[utx, ctx],
        t_start=t_enc,
        unconditional_guidance_scale=scale,
        xtype='image',
        first_ctype='vision',
        second_ctype='prompt',
        mixed_ratio=(1 - mixing),
    )
    print(f'[{im_id}] decode_dc (dual-conditioning DDIM sampling) OK, output shape {z.shape}')

    z = z.to(DEVICE).half()
    x = net.autokl_decode(z)
    x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
    x = [tvtrans.ToPILImage()(xi) for xi in x]

    out_path = f'smoke_test_output/vd_smoke_{im_id}.png'
    x[0].save(out_path)
    assert os.path.getsize(out_path) > 0
    print(f'[{im_id}] saved {out_path}')

print('SMOKE TEST PASSED: Versatile Diffusion checkpoint loads and full dual-conditioning diffusion pipeline runs correctly.')
