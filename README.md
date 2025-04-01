# Iterative Window Mean Filter: Thwarting Diffusion-based Adversarial Purification

Hanrui Wang, Ruoxi Sun, Cunjian Chen, Minhui Xue, Lay-Ki Soon, Shuo Wang, Zhe Jin

[Publication](https://ieeexplore.ieee.org/abstract/document/10704070) | [PDF](https://arxiv.org/pdf/2408.10673)

Face authentication systems have brought significant convenience and advanced developments, yet they have become unreliable due to their sensitivity to inconspicuous perturbations, such as adversarial attacks. Existing defenses often exhibit weaknesses when facing various attack algorithms and adaptive attacks or compromise accuracy for enhanced security. To address these challenges, we have developed a novel and highly efficient non-deep-learning-based image filter called the Iterative Window Mean Filter (IWMF) and proposed a new framework for adversarial purification, named IWMF-Diff, which integrates IWMF and denoising diffusion models. These methods can function as pre-processing modules to eliminate adversarial perturbations without necessitating further modifications or retraining of the target system. We demonstrate that our proposed methodologies fulfill four critical requirements: preserved accuracy, improved security, generalizability to various threats in different settings, and better resistance to adaptive attacks. This performance surpasses that of the state-of-the-art adversarial purification method, DiffPure.

* This is also an unofficial but more efficient implemntation of [DiffPure](https://github.com/NVlabs/DiffPure) for face recognition, CIFAR-10, and ImageNet ( --lambda_0 0).

****
## Contents
* [Introduction](#Introduction)
* [Main Requirements](#Main-Requirements)
* [Installation](#Installation)
* [Data Preparation](#Data-Preparation)
* [Pretrained Models](#Pretrained-Models)
* [Usage](#Usage)
* [Results](#Results)
* [Citation](#Citation)
* [Acknowledgement](#Acknowledgement)
* [Contacts](#Contacts)

****

## Introduction
The procedure of IWMF defending the authentication system is as follows:
* Step 1: Blur the input image by IWMF. Perturbations on adversarial examples are largely removed, yet facial features are partially distorted.
* Step 2: Restore image by DDRM. Robustness against both genuine images and adversarial examples raises.
* Step 3: Verify the pre-processed image by a regular authentication system. Note that users do not need to re-enroll due to the defense.

<img src="figures/pipeline.jpg" alt="pipeline" style="width:500px;"/>

## Main Requirements

  * **Python (3.9.18)**
  * **torch (2.1.2+cu118)**
  * **torchvision (0.16.2+cu118)**
  * **PyYAML (6.0.1)**
  * **tqdm (4.66.2)**
  * **facenet-pytorch (2.5.3)**
  
  The versions in `()` have been tested.

<!-- ## Installation
```
git clone https://github.com/azrealwang/iwmfdiff.git
cd iwmfdiff
pip3 install -r requirements.txt
```
if equipped with GPU:

```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```
or:

```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
``` -->
## Installation

This guide assumes you have an empty Python environment (3.9+) activated. `iwmfdiff` supports both CPU and GPU (CUDA) usage.

#### Step 1: Clone the Repository

```bash
git clone https://github.com/azrealwang/iwmfdiff.git
cd iwmfdiff
```

#### Step 2: Install with CUDA (GPU) or CPU spoort

* For GPU (CUDA 11.8):

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install -e .
```

* For CPU Only:

```bash
pip install -e .
```

#### Step 3: Verify Installation

```bash
python -c "import torch; print(torch.__version__)"  # Should be 2.1.2+cu118 if pre-installed
python -c "import numpy; print(numpy.__version__)"  # Should be <2, e.g., 1.26.4
python -c "import iwmfdiff; print(iwmfdiff.__version__)"  # Should print: 0.1.0
python -c "import torch; print(torch.cuda.is_available())"  # True if CUDA installed
```

#### Notes

* CUDA requires a compatible toolkit (e.g., 11.8). Check with `nvcc --version`.
* If NumPy 2.x issues arise, ensure `numpy<2` is used (`pip install "numpy<2"`).

## Data Preparation

The image name must satisfy `00000_0.jpg`. `00000` and `_0` indicates the image id and user id/class/label, respectively. The image id must be unique and auto-increment from `00000`. `.jpg` can be any image file format. 

20 target and source images have been prepared in `imgs/` for running [demos](#Usage).

## Pretrained Models

* [InsightFace](https://github.com/deepinsight/insightface): iresnet100 pretrained using the CASIA dataset; automatically downloaded

* [FaceNet](https://github.com/timesler/facenet-pytorch): InceptionResnetV1 pretrained using the VGG2FACE dataset; automatically downloaded

* [Denoising diffusion models](https://github.com/bahjat-kawar/ddrm): pretrained using the CelebA-HQ dataset; automatically downloaded

Sometimes, the download speed of denoising diffusion models is very slow. Then, please manually download the pretrained model from [Google Drive](https://drive.google.com/file/d/1LeawZE7MKtrr0l-4_UOufiMykORy_MA1/view?usp=share_link) and prepare it as the path `exp/logs/celeba/celeba_hq.ckpt`.

## Usage

After installing the package (e.g., `pip install -e .`), you can use the provided command-line tools to perform attacks, purifications, and evaluations. These tools are available as console scripts, allowing you to run `iwmf-attack`, `iwmf-defense`, and `iwmf-eval` directly instead of invoking python with the script paths.

### Regular attack
```
iwmf-attack --attack APGD --norm Linf --eps 0.03 --model insightface --thres 0.6351
```
_(Previously: `python attack.py ...`)_
### Purify adversarial examples
```
iwmf-defense --lambda_0 0.25 --sigma_y 0.15 --folder APGD-Linf-0.03-insightface-0.6351 --input imgs/adv --eval_adv --model insightface --thres 0.6351
```
_(Previously: `python defense.py ...`)_
### Purify genuine images
```
iwmf-defense --lambda_0 0.25 --sigma_y 0.15 --folder target --input imgs --eval_genuine --model insightface --thres 0.6351
```
(Previously: `python defense.py ...`)_
### Adaptive attack
```
iwmf-attack --attack Adaptive --norm Linf --eps 0.03 --defense 0.25 0.15 --model insightface --thres 0.6351
```
_(Previously: `python attack.py ...`)_
where the following are partial options:
- `--model` allows `facenet` or `insightface`
- `--attack` allows `APGD` (white-box attack), `APGD_EOT` (adaptive attack), `Square` (black-box attack), or `Adaptive` (strong adaptive)
- `--eval_genuine` runs the task that computes FRR for genuine images before or after purification
- `--eval_adv` runs the task that computes FAR and FRR for adv before or after purification

Other options refer to `--help`

#### Notes

* __Console Scripts__: The commands `iwmf-attack` and `iwmf-defense` are provided via the `[project.scripts]` section in `pyproject.toml`. This simplifies usage by eliminating the need to prepend `python` and specify script paths. Ensure the package is installed (`pip install -e .`) to use these commands.

### Import for pre-processing
```
from functions.defense import iwmfdiff
```
```python
def iwmfdiff(
	imgs_input: Tensor,
	lambda_0: float,
	sigma_y: float,
	s: int = 3,
	batch: int = 1,
	seed: int = None,
	data: str = 'celeba_hq', # option for celeba_hq, cifar10, imagenet_256
	) -> Tensor:
```

### CIFAR-10 and ImageNet

The defense settings (i.e., lambda_0 and sigma_y) should to be determined accroding to trade-off between clean and robust accuracy.

If the downloading is slow, please manually download the pretrained model from [Google Drive](https://drive.google.com/file/d/1LeawZE7MKtrr0l-4_UOufiMykORy_MA1/view?usp=share_link) and prepare it as the path:
* CIFAR-10: `exp/logs/diffusion_models_converted/ema_diffusion_cifar10_model/model-790000.ckpt`
* ImageNet: `exp/logs/imagenet/256x256_diffusion_uncond.pt`

## Results (%)

| Defense | FRR-Genuie | FAR-APGD | FAR-APGD_EOT | FAR-Square | FAR-Adaptive | Time Cost (s) |
|:---:|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Insightface | 0.28 | 100 | 100 | 100 | N/A | N/A |
| DiffPure | 5.00 | 17.4 | 17.6 | 20.4 | 99.4 | 3.41 |
| IWMF | 6.36 | 9.2 | 7.6 | 28.8 | 80.4 | 0.36 |
| IWMF-Diff | 3.22 | 6.6 | 5.0 | 19.8 | 77.4 | 3.79 |

## Citation
```
@article{wang2024iterative,
  title={Iterative Window Mean Filter: Thwarting Diffusion-based Adversarial Purification},
  author={Wang, Hanrui and Sun, Ruoxi and Chen, Cunjian and Xue, Minhui and Soon, Lay-Ki and Wang, Shuo and Jin, Zhe},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2024},
  publisher={IEEE}
}
```

## Acknowledgement
The implementation is partially inspired by:
* [AutoAttack](https://github.com/fra31/auto-attack) (adversarial examples generation)
* [DDRM](https://github.com/bahjat-kawar/ddrm) (image restoration)

## Contacts
If you have any questions about our work, please do not hesitate to contact us by email.

Hanrui Wang: hanrui_wang@nii.ac.jp

