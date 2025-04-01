import os
from pathlib import Path
from importlib.resources import files
import numpy as np
import math
from typing import Optional, Union
import torch
import yaml
from torch import Tensor
from .utils import imgs_resize, save_all_images

from .ddrm.functions.denoising import efficient_generalized_steps, get_beta_schedule, dict2namespace
from .ddrm.functions.ckpt_util import get_ckpt_path, download
from .ddrm.models import diffusion
from .ddrm.guided_diffusion.script_util import create_model

# Define package root (the iwmfdiff package directory)
PACKAGE_ROOT = Path(__file__).parent  # Points to iwmfdiff/iwmfdiff
# print("PACKAGE_ROOT:", PACKAGE_ROOT)  # Debug: verify package root
CONFIG_DIR = PACKAGE_ROOT / "configs"
# print("CONFIG_DIR:", CONFIG_DIR)      # Debug: verify config directory
EXP_DIR = PACKAGE_ROOT / "exp"
# # Uncomment the following lines to use a different cache directory
# # Use user cache directory
# CACHE_DIR = Path.home() / ".cache" / "iwmfdiff"
# EXP_DIR = CACHE_DIR / "exp"
EXP_DIR.mkdir(exist_ok=True)          # Ensure exp directory exists
# print("EXP_DIR:", EXP_DIR)            # Debug: verify exp directory


def obtain_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, torch.device):
        pass
    else:
        raise ValueError("device must be None, str, or torch.device")
    return device


def iwmfdiff(
        imgs_input: Tensor,
        lambda_0: float,
        sigma_y: float,
        s: int = 3,
        batch: int = 1,
        seed: int = None,
        data: str = 'celeba_hq',  # option for celeba_hq, cifar10, imagenet_256
        timesteps: int = 20,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = False,
        ) -> Tensor:
    if seed is not None:
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.random.manual_seed(seed)
    _, _, h, w = imgs_input.shape
    imgs = iwmf(imgs_input, lambda_0, s)
    device = obtain_device(device)
    imgs = imgs.to(device)
    if sigma_y > 0:
        if verbose:
            print("**********************denoising images******************************")
            print("sigma_y: ", sigma_y)
        if data == 'cifar10':
            imgs = imgs_resize(imgs,(32,32))
        else:
            imgs = imgs_resize(imgs,(256,256))
        imgs = ddrm(imgs, batch=batch, sigma_0=sigma_y, data=data, timesteps=timesteps, device=device, verbose=verbose)
        imgs = Tensor(imgs_resize(imgs,(h,w))).to(device)
    
    return imgs

def iwmf(
        img_input: Tensor,
        lambda_0: float,
        s: int = 3,
        ) -> Tensor:
    img_output = img_input.clone()
    i, c, h, w = img_input.shape
    s_left = math.floor(s/2)
    s_right = math.ceil(s/2)
    for _ in range(int(lambda_0*h*w)):
        x = np.random.randint(s_left,h-s_left)
        y = np.random.randint(s_left,w-s_left)
        array_conv = img_output[:,:,x-s_left:x+s_right,y-s_left:y+s_right].reshape(i,c,s*s)
        fixed_value = torch.mean(array_conv,dim=2,keepdim=True)
        img_output[:,:,x-s_left:x+s_right,y-s_left:y+s_right] = fixed_value.unsqueeze(3).repeat(1,1,s,s)
    
    return img_output


def ddrm(
        img: Tensor,
        timesteps: int = 20,  # default 20
        sigma_0: float = 0.1,  # default 0.1
        batch: int = 1,
        # config_name: str = "celeba_hq.yml",
        exp: str = "exp",  # deprecated
        deg: str = "deno",
        data: str = 'celeba_hq',
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = False,
        ) -> Tensor:
        # load config
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = obtain_device(device)
        if data == 'celeba_hq':
            config_name: str = "celeba_hq.yml"
        elif data == 'cifar10':
            config_name: str = "cifar10.yml"
        elif data == 'imagenet_256':
            config_name: str = "imagenet_256.yml"
        else:
            raise ValueError
        img = img.to(device)
        # with open(os.path.join("configs", config_name), "r") as f:
        config_path = CONFIG_DIR / config_name
        with config_path.open('r') as f:
            config_tmp = yaml.safe_load(f)
        config = dict2namespace(config_tmp)
        config.device = device
        
        #assert img.shape[2] == config.data.image_size
        
        # load betas
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float().to(device)
        
        # other parameters
        num_timesteps = betas.shape[0]
        skip = num_timesteps // timesteps
        seq = range(0, num_timesteps, skip)
        
        # load model
        if config.model.type == 'simple':    
            model = diffusion.Model(config)
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif config.data.dataset == "LSUN":
                name = f"lsun_{config.data.category}"
            elif config.data.dataset == 'CelebA_HQ':
                name = 'celeba_hq'
            else:
                raise ValueError
            if name != 'celeba_hq':
                ckpt = get_ckpt_path(f"ema_{name}", prefix=str(EXP_DIR))
                if verbose:
                    print("Loading checkpoint {}".format(ckpt))
            elif name == 'celeba_hq':
                #ckpt = '~/.cache/diffusion_models_converted/celeba_hq.ckpt'
                # ckpt = os.path.join(exp, "logs/celeba/celeba_hq.ckpt")
                ckpt = EXP_DIR / "logs/celeba/celeba_hq.ckpt"
                # if not os.path.exists(ckpt):
                if not ckpt.exists():
                    ckpt.parent.mkdir(parents=True, exist_ok=True)
                    ckpt_url = 'https://huggingface.co/gwang-kim/DiffusionCLIP-CelebA_HQ/resolve/main/celeba_hq.ckpt'
                    # ckpt_url = 'https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt'
                    download(ckpt_url, ckpt)

            else:
                raise ValueError

            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.to(device)
            if device!="cpu":
                model = torch.nn.DataParallel(model)

        elif config.model.type == 'openai':
            config_dict = vars(config.model)
            model = create_model(**config_dict)
            if config.model.use_fp16:
                model.convert_to_fp16()
            if config.model.class_cond:
                # ckpt = os.path.join(exp, 'logs/imagenet/%dx%d_diffusion.pt' % (config.data.image_size, config.data.image_size))
                ckpt = EXP_DIR / f"logs/imagenet/{config.data.image_size}x{config.data.image_size}_diffusion.pt"
                # if not os.path.exists(ckpt):
                if not ckpt.exists():
                    ckpt.parent.mkdir(parents=True, exist_ok=True)
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (config.data.image_size, config.data.image_size), ckpt)
            else:
                # ckpt = os.path.join(exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                ckpt = EXP_DIR / "logs/imagenet/256x256_diffusion_uncond.pt"
                # if not os.path.exists(ckpt):
                if not ckpt.exists():
                    ckpt.parent.mkdir(parents=True, exist_ok=True)
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', ckpt)
                
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.to(device)
            model.eval()
            if device!="cpu":
                model = torch.nn.DataParallel(model)
            
            
        # load deg
        H_funcs = None
        if deg[:2] == 'cs':
            compress_by = int(deg[2:])
            from .ddrm.functions.svd_replacement import WalshHadamardCS
            H_funcs = WalshHadamardCS(config.data.channels, config.data.image_size, compress_by, torch.randperm(config.data.image_size**2, device=device), device)
        elif deg[:3] == 'inp':
            from .ddrm.functions.svd_replacement import Inpainting
            if deg == 'inp_lolcat':
                loaded = np.load("inp_masks/lolcat_extra.npy")
                mask = torch.from_numpy(loaded).to(device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            elif deg == 'inp_lorem':
                loaded = np.load("inp_masks/lorem3.npy")
                mask = torch.from_numpy(loaded).to(device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            else:
                missing_r = torch.randperm(config.data.image_size**2)[:config.data.image_size**2 // 2].to(device).long() * 3
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
            H_funcs = Inpainting(config.data.channels, config.data.image_size, missing, device)
        elif deg == 'deno':
            from .ddrm.functions.svd_replacement import Denoising
            H_funcs = Denoising(config.data.channels, config.data.image_size, device)
        elif deg[:10] == 'sr_bicubic':
            factor = int(deg[10:])
            from .ddrm.functions.svd_replacement import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1/factor)*(i- np.floor(factor*4/2) +0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(device)
            H_funcs = SRConv(kernel / kernel.sum(), \
                             config.data.channels, config.data.image_size, device, stride = factor)
        elif deg == 'deblur_uni':
            from .ddrm.functions.svd_replacement import Deblurring
            H_funcs = Deblurring(torch.Tensor([1/9] * 9).to(device), config.data.channels, config.data.image_size, device)
        elif deg == 'deblur_gauss':
            from .ddrm.functions.svd_replacement import Deblurring
            sigma = 10
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device)
            H_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, config.data.image_size, device)
        elif deg == 'deblur_aniso':
            from .ddrm.functions.svd_replacement import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)
            H_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels, config.data.image_size, device)
        elif deg[:2] == 'sr':
            blur_by = int(deg[2:])
            from .ddrm.functions.svd_replacement import SuperResolution
            H_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, device)
        elif deg == 'color':
            from .ddrm.functions.svd_replacement import Colorization
            H_funcs = Colorization(config.data.image_size, device)
        else:
            print("ERROR: degradation type not supported")
            quit()
        
        # initial x 
        l = img.shape[0]
        x = torch.randn(
            l,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=device,
            )
        
        # process denoising in batch
        for i in range(math.ceil(l/batch)):
            if i == math.ceil(l/batch)-1:
                x[i*batch:l], _ = efficient_generalized_steps(x[i*batch:l], seq, model, betas, H_funcs, img[i*batch:l], sigma_0=sigma_0, etaB=1, etaA=0.85, etaC=0.85, cls_fn=None, classes=None, verbose=verbose)
            else:
                x[i*batch:(i+1)*batch], _ = efficient_generalized_steps(x[i*batch:(i+1)*batch], seq, model, betas, H_funcs, img[i*batch:(i+1)*batch], sigma_0=sigma_0, etaB=1, etaA=0.85, etaC=0.85, cls_fn=None, classes=None, verbose=verbose)
        
        return x