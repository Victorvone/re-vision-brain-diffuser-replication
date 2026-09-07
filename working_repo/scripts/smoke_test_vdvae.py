"""
Smoke test: confirms the vendored VDVAE code + downloaded ImageNet64 checkpoint
still load and run end-to-end under the current environment (Python 3.12,
PyTorch 2.12+ROCm), using synthetic images/latents instead of real NSD data.
Mirrors scripts/vdvae_reconstruct_images.py's exact code path.
"""
import sys
sys.path.append('vdvae')
import torch
import numpy as np
from data import mkdir_p
from vae import VAE
from train_helpers import restore_params
from image_utils import *
from model_utils import *
from PIL import Image
import torchvision.transforms as T
import os

print('Libs imported OK')

H = {'image_size': 64, 'image_channels': 3, 'seed': 0, 'port': 29500, 'save_dir': './saved_models/test',
     'data_root': './', 'desc': 'test', 'hparam_sets': 'imagenet64',
     'restore_path': 'imagenet64-iter-1600000-model.th',
     'restore_ema_path': 'vdvae/model/imagenet64-iter-1600000-model-ema.th',
     'restore_log_path': 'imagenet64-iter-1600000-log.jsonl',
     'restore_optimizer_path': 'imagenet64-iter-1600000-opt.th', 'dataset': 'imagenet64', 'ema_rate': 0.999,
     'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5',
     'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12', 'zdim': 16, 'width': 512,
     'custom_width_str': '', 'bottleneck_multiple': 0.25, 'no_bias_above': 64, 'scale_encblock': False,
     'test_eval': True, 'warmup_iters': 100, 'num_mixtures': 10, 'grad_clip': 220.0, 'skip_threshold': 380.0,
     'lr': 0.00015, 'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 'n_batch': 4,
     'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0, 'iters_per_ckpt': 25000, 'iters_per_print': 1000,
     'iters_per_save': 10000, 'iters_per_images': 10000, 'epochs_per_eval': 1, 'epochs_per_probe': None,
     'epochs_per_eval_save': 1, 'num_images_visualize': 8, 'num_variables_visualize': 6,
     'num_temperatures_visualize': 3, 'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'}


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


H = dotdict(H)

from data import set_up_data
# NOTE: set_up_data() actually returns 4 values (H, train_data, valid_data,
# preprocess_func); brain-diffuser's own scripts unpack it as `H, preprocess_fn =
# set_up_data(H)`, which is a bug in the published repo (crashes with a
# ValueError as-is). Fixed here and in the three real vdvae_*.py scripts.
H, _, _, preprocess_fn = set_up_data(H)
print('set_up_data OK')

print('Loading ema VAE checkpoint (critical test: old .th checkpoint under new torch)...')
ema_vae = load_vaes(H)
print('load_vaes OK — checkpoint loaded successfully')

# Synthetic "test images" (no real NSD data needed for this smoke test)
n_test = 2
rng = np.random.default_rng(0)
synthetic_images = rng.integers(0, 256, size=(n_test, 64, 64, 3), dtype=np.uint8)

batch = []
for i in range(n_test):
    img = Image.fromarray(synthetic_images[i])
    img = T.functional.resize(img, (64, 64))
    img = torch.tensor(np.array(img)).float()
    batch.append(img)
x = torch.stack(batch)

data_input, target = preprocess_fn([x])  # bug fix: preprocess_fn expects x[0] to be the batch tensor
with torch.no_grad():
    activations = ema_vae.encoder.forward(data_input)
    px_z, stats = ema_vae.decoder.forward(activations, get_latents=True)
print(f'Full encoder+decoder forward pass OK — {len(stats)} latent layers total, first layer z shape {stats[0]["z"].shape}')
# The full VDVAE hierarchy has 75 layers (paper's own methods text); brain-diffuser's
# code deliberately only uses the first 31 for regression, so we only need 31 available.
assert len(stats) >= 31, f"expected at least 31 latent layers, got {len(stats)}"

# Mirror the real reconstruction script: build synthetic 31-layer "predicted" latents,
# reshape via the same layer_dims scheme, and decode via forward_manual_latents.
layer_dims = np.array([2**4, 2**4, 2**8, 2**8, 2**8, 2**8, 2**10, 2**10, 2**10, 2**10, 2**10, 2**10, 2**10, 2**10,
                        2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12,
                        2**12, 2**12, 2**12, 2**14])
total_dim = int(layer_dims.sum())
pred_latents = rng.standard_normal((n_test, total_dim)).astype(np.float32)


def latent_transformation(latents, ref):
    transformed_latents = []
    for i in range(31):
        t_lat = latents[:, layer_dims[:i].sum():layer_dims[:i + 1].sum()]
        c, h, w = ref[i]['z'].shape[1:]
        transformed_latents.append(t_lat.reshape(len(latents), c, h, w))
    return transformed_latents


input_latent = latent_transformation(pred_latents, stats)


def sample_from_hier_latents(latents, sample_ids):
    sample_ids = [sid for sid in sample_ids if sid < len(latents[0])]
    return [torch.tensor(latents[i][sample_ids]).float().cuda() for i in range(len(latents))]


samp = sample_from_hier_latents(input_latent, range(n_test))
with torch.no_grad():
    px_z = ema_vae.decoder.forward_manual_latents(len(samp[0]), samp, t=None)
    sample_from_latent = ema_vae.decoder.out_net.sample(px_z)
print(f'forward_manual_latents + out_net.sample OK — output shape {np.array(sample_from_latent).shape}')

out_dir = 'smoke_test_output'
mkdir_p(out_dir)
for j in range(len(sample_from_latent)):
    im = Image.fromarray(sample_from_latent[j])
    im = im.resize((512, 512), resample=3)
    im.save(f'{out_dir}/vdvae_smoke_{j}.png')
    assert os.path.getsize(f'{out_dir}/vdvae_smoke_{j}.png') > 0

print('SMOKE TEST PASSED: VDVAE checkpoint loads and full encode/decode/reconstruct pipeline runs correctly.')
