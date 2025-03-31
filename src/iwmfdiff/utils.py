import os
import numpy as np
import torch
from typing import Optional, Tuple, Any
from torch.nn import CosineSimilarity
from torch import Tensor
from torchvision.utils import save_image, make_grid
from PIL import Image
from math import ceil

# def imgs_resize(
#         imgs: Tensor,
#         shape: Tuple[int, int],
#         ) -> Tensor:
#     import torch.nn.functional as F
#     images = F.interpolate(imgs, size=shape, mode='bilinear', align_corners=False)
    
#     return images

def imgs_resize(
        imgs: Tensor,
        shape: Tuple[int, int],
        ) -> Tensor:
    from torchvision.transforms import Resize
    transform = Resize(shape,antialias=True)
    images = transform(imgs)
    
    return images

# def imgs_resize(imgs,shape):
#     l,_,_,_ = imgs.shape
#     images = []
#     for idx in range(l):
#         grid = make_grid(imgs[idx])
#         ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
#         im = Image.fromarray(ndarr)
#         im = im.resize(shape)
#         image = np.asarray(im, dtype=np.float32)
#         image = np.transpose(image, (2, 0, 1))
#         images.append(image)
#     images = Tensor(np.divide(images,255))
    
#     return images

def save_all_images(
        imgs: Any,
        labels: Any,
        output_path: str,
        start_idx: Optional[int] = 0,
        ) -> None:
    amount = imgs.shape[0]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in range(amount):
        save_image(imgs[i], f'{output_path}/%05d_%d.png'%(i+start_idx,labels[i]))

def cos_similarity_score(featuresA: Tensor, featuresB: Tensor) -> Tensor:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cos = CosineSimilarity(dim=1,eps=1e-6)
    featuresA = featuresA.to(device)
    featuresB = featuresB.to(device)
    similarity = (cos(featuresA,featuresB)+1)/2
    del featuresA, featuresB
    
    return similarity

def load_samples(
    path: str,
    amount: Optional[int] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> Tuple[Any, Any]:   

    images, labels = [], []
    basepath = r""
    samplepath = os.path.join(basepath, f"{path}")
    files = os.listdir(path)
    if amount is None:
        amount = len(files)
    for i in range(amount):
        # get filename and label
        file = [n for n in files if f"{i:05d}_" in n][0]
        label = int(file.split(".")[0].split("_")[-1])

        # open file
        path = os.path.join(samplepath, file)
        image = Image.open(path)
        
        if shape is not None:
            image = image.resize(shape)
        
        image = np.asarray(image, dtype=np.float32)

        if image.ndim == 2:
            image = image[..., np.newaxis]

        assert image.ndim == 3
        
        image = np.transpose(image, (2, 0, 1))
        
        images.append(image)
        labels.append(label)
    
    images_ = np.stack(images)
    labels_ = np.array(labels)

    images_ = images_ / 255
    images_ = np.ascontiguousarray(images_)
    
    return images_, labels_


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

def FMR(
    advs: Tensor, 
    targets: Tensor, 
    thresh: float,
) -> Tuple[Any, Any]:
    scores = cos_similarity_score(advs,targets)
    fmr = -1
    if len(scores)>0:
        fmr = np.float32(scores.cpu()>=thresh).mean()

    return fmr

def predict(
    model,
    imgs: Tensor,
    batch_size: int=1,
    device: str='cpu',
    )-> Tensor:
    count = len(imgs)
    batches = ceil(count/batch_size)
    logits = Tensor([])
    for b in range(batches):
        if b == batches-1:
            idx = range(b*batch_size,count)
        else:
            idx = range(b*batch_size,(b+1)*batch_size)
        with torch.no_grad():
            logits_batch = model.to(device)(imgs[idx].to(device)).cpu()
        logits = torch.cat((logits, logits_batch), 0)
    
    return logits