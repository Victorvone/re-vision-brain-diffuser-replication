import sys
sys.path.append('versatile_diffusion')
import os
import os.path as osp
import PIL
from PIL import Image
from pathlib import Path
import numpy as np
import numpy.random as npr

import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from torch.utils.data import DataLoader, Dataset

from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import matplotlib.pyplot as plt
from skimage.transform import resize, downscale_local_mean

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-diff_str", "--diff_str",help="Diffusion Strength",default=0.75)
parser.add_argument("-mix_str", "--mix_str",help="Mixing Strength",default=0.4)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]
strength = float(args.diff_str)
mixing = float(args.mix_str)


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

        assert (x.shape[1]==512) & (x.shape[2]==512), \
            'Wrong image size'
        return x

DEVICE = 'cuda:0'

cfgm_name = 'vd_noema'
sampler = DDIMSampler_VD
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)

# get_model() builds the 3.3B-param model in fp32 by default. On this machine (16GB
# cgroup-capped unified memory, no swap) that alone peaks CPU RAM at ~17GB -- already
# over budget before any checkpoint or .half() cast gets a chance to run. GPU device
# memory is NOT charged against the cgroup limit here, so the fix is to never
# materialize the model in CPU RAM at all: build it directly on cuda via the
# torch.device context manager (requires the `accelerate` package so transformers'
# CLIPModel.from_pretrained honors the device context, and the CPU-forced fix in
# diffusion_utils.make_beta_schedule for its internal .numpy() call). See
# scripts/smoke_test_versatile_diffusion.py, which validates this end-to-end.
with torch.device(DEVICE):
    net = get_model()(cfgm)
net.half()
sd = torch.load(pth, map_location=DEVICE)
net.load_state_dict(sd, strict=False)
del sd

# Single-GPU consolidation: this machine has one GPU, not the two 12GB discrete GPUs
# the original code assumes (cuda(0)/cuda(1) split) -- everything lives on DEVICE.
sampler = sampler(net)
batch_size = 1

pred_text = np.load('data/predicted_features/subj{:02d}/nsd_cliptext_predtest_nsdgeneral.npy'.format(sub))
pred_text = torch.tensor(pred_text).half().to(DEVICE)

pred_vision = np.load('data/predicted_features/subj{:02d}/nsd_clipvision_predtest_nsdgeneral.npy'.format(sub))
pred_vision = torch.tensor(pred_vision).half().to(DEVICE)


n_samples = 1
ddim_steps = 50
ddim_eta = 0
scale = 7.5
xtype = 'image'
ctype = 'prompt'

torch.manual_seed(0)
for im_id in range(len(pred_vision)):

    zim = Image.open('results/vdvae/subj{:02d}/{}.png'.format(sub,im_id))

    zim = regularize_image(zim)
    zin = zim*2 - 1
    zin = zin.unsqueeze(0).to(DEVICE).half()

    init_latent = net.autokl_encode(zin)

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    #strength=0.75
    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(DEVICE))

    dummy = ''
    utx = net.clip_encode_text(dummy)
    utx = utx.to(DEVICE).half()

    dummy = torch.zeros((1,3,224,224)).to(DEVICE)
    uim = net.clip_encode_vision(dummy)
    uim = uim.to(DEVICE).half()

    z_enc = z_enc.to(DEVICE)

    h, w = 512,512
    shape = [n_samples, 4, h//8, w//8]

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
        mixed_ratio=(1-mixing), )

    z = z.to(DEVICE).half()
    x = net.autokl_decode(z)
    color_adj='None'
    #color_adj_to = cin[0]
    color_adj_flag = (color_adj!='none') and (color_adj!='None') and (color_adj is not None)
    color_adj_simple = (color_adj=='Simple') or color_adj=='simple'
    color_adj_keep_ratio = 0.5
    
    if color_adj_flag and (ctype=='vision'):
        x_adj = []
        for xi in x:
            color_adj_f = color_adjust(ref_from=(xi+1)/2, ref_to=color_adj_to)
            xi_adj = color_adj_f((xi+1)/2, keep=color_adj_keep_ratio, simple=color_adj_simple)
            x_adj.append(xi_adj)
        x = x_adj
    else:
        x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
        x = [tvtrans.ToPILImage()(xi) for xi in x]
    

    x[0].save('results/versatile_diffusion/subj{:02d}/{}.png'.format(sub,im_id))
      

