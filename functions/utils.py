import eagerpy as ep
import warnings
import os
import numpy as np
import torch
from typing import Optional, Tuple, Any
from torch.nn import CosineSimilarity
from torch import Tensor
from torchvision.utils import save_image, make_grid
from PIL import Image
from .types import Bounds
from .models import Model

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

def imgs_resize(imgs,shape):
    l,_,_,_ = imgs.shape
    images = []
    for idx in range(l):
        grid = make_grid(imgs[idx])
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im = im.resize(shape)
        image = np.asarray(im, dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))
        images.append(image)
    images = Tensor(np.divide(images,255))
    
    return images

def save_all_images(
        imgs: Any,
        labels: Any,
        output_path: str
        ) -> None:
    amount = imgs.shape[0]
    for i in range(amount):
        save_image(imgs[i], f'{output_path}/%05d_%d.png'%(i,labels[i]))

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