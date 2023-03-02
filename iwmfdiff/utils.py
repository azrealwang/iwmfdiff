from typing import Optional, Tuple, Any
import eagerpy as ep
import warnings
import os
import numpy as np
import math
import torch
import yaml
from torch.nn import CosineSimilarity
from torch import Tensor
from .types import Bounds
from .models import Model
from torchvision.utils import save_image
from PIL import Image

from ddrm.functions.denoising import efficient_generalized_steps, get_beta_schedule, dict2namespace
from ddrm.functions.ckpt_util import get_ckpt_path, download
from ddrm.models import diffusion
from ddrm.guided_diffusion.script_util import create_model

def samples(
    fmodel: Model,
    dataset: str = "lfw",
    index: int = 0,
    batchsize: int = 1,
    data_format: Optional[str] = None,
    bounds: Optional[Bounds] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> Any:
    if hasattr(fmodel, "data_format"):
        if data_format is None:
            data_format = fmodel.data_format  # type: ignore
        elif data_format != fmodel.data_format:  # type: ignore
            raise ValueError(
                f"data_format ({data_format}) does not match model.data_format ({fmodel.data_format})"  # type: ignore
            )
    elif data_format is None:
        raise ValueError(
            "data_format could not be inferred, please specify it explicitly"
        )

    if bounds is None:
        bounds = fmodel.bounds

    images, labels = _samples(
        dataset=dataset,
        index=index,
        batchsize=batchsize,
        data_format=data_format,
        bounds=bounds,
        shape=shape,
    )
    
    if hasattr(fmodel, "dummy") and fmodel.dummy is not None:  # type: ignore
        images = ep.from_numpy(fmodel.dummy, images).raw  # type: ignore
        labels = ep.from_numpy(fmodel.dummy, labels).raw  # type: ignore
    else:
        warnings.warn(f"unknown model type {type(fmodel)}, returning NumPy arrays")
    return images, labels


def _samples(
    dataset: str,
    index: int,
    batchsize: int,
    data_format: str,
    bounds: Bounds,
    shape: Tuple[int, int],
) -> Tuple[Any, Any]:   

    images, labels = [], []
    basepath = r""
    samplepath = os.path.join(basepath, f"{dataset}")
    files = os.listdir(samplepath)
    for idx in range(index, index + batchsize):
        i = idx

        # get filename and label
        file = [n for n in files if f"{i:05d}_" in n][0]
        label = int(file.split(".")[0].split("_")[-1])

        # open file
        path = os.path.join(samplepath, file)
        image = Image.open(path)
        
        # if model_type == "insightface" or model_type == "CurricularFace":
        #     image = image.resize((112, 112))
        if shape is not None:
            image = image.resize(shape)
        
        image = np.asarray(image, dtype=np.float32)

        if image.ndim == 2:
            image = image[..., np.newaxis]

        assert image.ndim == 3

        if data_format == "channels_first":
            image = np.transpose(image, (2, 0, 1))
        
        images.append(image)
        labels.append(label)
    
    images_ = np.stack(images)
    labels_ = np.array(labels)

    if bounds != (0, 255):
        images_ = images_ / 255 * (bounds[1] - bounds[0]) + bounds[0]
    return images_, labels_

def cos_similarity_score(featuresA: Tensor, featuresB: Tensor) -> Tensor:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cos = CosineSimilarity(dim=1,eps=1e-6)
    featuresA = featuresA.to(device)
    featuresB = featuresB.to(device)
    similarity = (cos(featuresA,featuresB)+1)/2
    del featuresA, featuresB
    return similarity

def false_rate(
    featuresA: Tensor, 
    labelsA: Tensor, 
    featuresB: Tensor, 
    lablesB: Tensor,
    thresh: float,
) -> Tuple[Any, Any]:
    geniue_indexA = list()
    geniue_indexB = list()
    imposter_indexA = list()
    imposter_indexB = list()
    for i in range(labelsA.shape[0]):
        for j in range(lablesB.shape[0]):
            if labelsA[i]==lablesB[j]:
                geniue_indexA.extend([i])
                geniue_indexB.extend([j])
            else:
                imposter_indexA.extend([i])
                imposter_indexB.extend([j])
    geniue_score = cos_similarity_score(featuresA[geniue_indexA],featuresB[geniue_indexB])
    imposter_score = cos_similarity_score(featuresA[imposter_indexA],featuresB[imposter_indexB])
    frr = -1
    if len(geniue_score)>0:
        frr = float(torch.Tensor.float(geniue_score < thresh).mean())
    far = -1
    if len(imposter_score)>0:
        far = float(torch.Tensor.float(imposter_score >= thresh).mean())
    return far, frr
    
def save_all_images(
        imgs: Any,
        subject: int,
        samplesize: int,
        output_path: str
        ) -> None:
    for i in range(subject):
        for j in range(samplesize):
            save_image(imgs[i*samplesize+j], f'{output_path}/%05d_%d.png'%(i*samplesize+j,i))

def FMR(
    advs: Tensor, 
    targets: Tensor, 
    thresh: float,
    samplesize: int,
) -> Tuple[Any, Any]:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    imposter_score_target = cos_similarity_score(advs,targets)
    imposter_score_renew = Tensor([]).to(device)
    totalsize = np.int(targets.shape[0])
    for i in range(samplesize):
        index_i = list(range(0+i,totalsize,samplesize))
        for j in range(samplesize): 
            if i!=j:
                index_j = list(range(0+j,totalsize,samplesize))
                similarity_tmp = cos_similarity_score(advs[index_i],targets[index_j])
                imposter_score_renew = torch.cat((imposter_score_renew,similarity_tmp),0)
    fmr_target = -1
    if len(imposter_score_target)>0:
        fmr_target = np.float32(imposter_score_target.cpu() >= thresh).mean()
    fmr_renew = -1
    if len(imposter_score_renew)>0:
        fmr_renew = np.float32(imposter_score_renew.cpu() >= thresh).mean()
    del imposter_score_target, imposter_score_renew

    return fmr_target, fmr_renew

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
        timesteps: int = 20, # default 20
        sigma_0: float = 0.1, # default 0.1
        batch: int = 1,
        config_name: str = "celeba_hq.yml",
        exp: str = "exp",
        deg: str = "deno",
        ) -> Tensor:
        # load config
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        img = img.to(device)
        with open(os.path.join("configs", config_name), "r") as f:
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
                ckpt = get_ckpt_path(f"ema_{name}", prefix=exp)
                print("Loading checkpoint {}".format(ckpt))
            elif name == 'celeba_hq':
                #ckpt = '~/.cache/diffusion_models_converted/celeba_hq.ckpt'
                ckpt = os.path.join(exp, "logs/celeba/celeba_hq.ckpt")
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt', ckpt)
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
                ckpt = os.path.join(exp, 'logs/imagenet/%dx%d_diffusion.pt' % (config.data.image_size, config.data.image_size))
                if not os.path.exists(ckpt):
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (config.data.image_size, config.data.image_size), ckpt)
            else:
                ckpt = os.path.join(exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
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
            from ddrm.functions.svd_replacement import WalshHadamardCS
            H_funcs = WalshHadamardCS(config.data.channels, config.data.image_size, compress_by, torch.randperm(config.data.image_size**2, device=device), device)
        elif deg[:3] == 'inp':
            from ddrm.functions.svd_replacement import Inpainting
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
            from ddrm.functions.svd_replacement import Denoising
            H_funcs = Denoising(config.data.channels, config.data.image_size, device)
        elif deg[:10] == 'sr_bicubic':
            factor = int(deg[10:])
            from ddrm.functions.svd_replacement import SRConv
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
            from ddrm.functions.svd_replacement import Deblurring
            H_funcs = Deblurring(torch.Tensor([1/9] * 9).to(device), config.data.channels, config.data.image_size, device)
        elif deg == 'deblur_gauss':
            from ddrm.functions.svd_replacement import Deblurring
            sigma = 10
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device)
            H_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, config.data.image_size, device)
        elif deg == 'deblur_aniso':
            from ddrm.functions.svd_replacement import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)
            H_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels, config.data.image_size, device)
        elif deg[:2] == 'sr':
            blur_by = int(deg[2:])
            from ddrm.functions.svd_replacement import SuperResolution
            H_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, device)
        elif deg == 'color':
            from ddrm.functions.svd_replacement import Colorization
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
                x[i*batch:l], _ = efficient_generalized_steps(x[i*batch:l], seq, model, betas, H_funcs, img[i*batch:l], sigma_0=sigma_0, etaB=1, etaA=0.85, etaC=0.85, cls_fn=None, classes=None)
            else:
                x[i*batch:(i+1)*batch],_ = efficient_generalized_steps(x[i*batch:(i+1)*batch], seq, model, betas, H_funcs, img[i*batch:(i+1)*batch], sigma_0=sigma_0, etaB=1, etaA=0.85, etaC=0.85, cls_fn=None, classes=None)
        
        return x