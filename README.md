# Iterative Window Mean Filter: Thwarting Diffusion-based Adversarial Purification

Hanrui Wang, Ruoxi Sun, Cunjian Chen, Minhui Xue, Lay-Ki Soon, Shuo Wang, Zhe Jin

Face authentication systems have brought significant convenience and advanced developments, yet they have become unreliable due to their sensitivity to inconspicuous perturbations, such as adversarial attacks. Existing defenses often exhibit weaknesses when facing various attack algorithms and adaptive attacks or compromise accuracy for enhanced security. To address these challenges, we have developed a novel and highly efficient non-deep-learning-based image filter called the Iterative Window Mean Filter (IWMF) and proposed a new framework for adversarial purification, named IWMF-Diff, which integrates IWMF and denoising diffusion models. These methods can function as pre-processing modules to eliminate adversarial perturbations without necessitating further modifications or retraining of the target system. We demonstrate that our proposed methodologies fulfill four critical requirements: preserved accuracy, improved security, generalizability to various threats in different settings, and better resistance to adaptive attacks. This performance surpasses that of the state-of-the-art adversarial purification method, DiffPure.

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

## Installation
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
```
## Data Preparation

The image name must satisfy `00000_0.jpg`. `00000` and `_0` indicates the image id and user id/class/label, respectively. The image id must be unique and auto-increment from `00000`. `.jpg` can be any image file format. 

20 target and source images have been prepared in `imgs` for running [demos](#Usage).

## Pretrained Models

* [InsightFace](https://github.com/deepinsight/insightface): iresnet100 pretrained using the CASIA dataset; automatically downloaded

* [FaceNet](https://github.com/timesler/facenet-pytorch): InceptionResnetV1 pretrained using the VGG2FACE dataset; automatically downloaded

* [Denoising diffusion models](https://github.com/bahjat-kawar/ddrm): pretrained using the CelebA-HQ dataset; automatically downloaded

Sometimes, the download speed of denoising diffusion models is very slow. Then, please manually download the pretrained model from [Google Drive](https://drive.google.com/file/d/1ulkO2GFepl1IRlPjMRS_vsaVq5wG0p_x/view?usp=share_link) and prepare it as the path `exp/logs/celeba/celeba_hq.ckpt`.

## Usage
### Regular attack
```
python attack.py --attack APGD --norm Linf --eps 0.03 --model insightface --thres 0.6351
```
### Purify adversarial examples
```
python defense.py --lambda_0 0.25 --sigma_y 0.15 --folder APGD-Linf-0.03-insightface-0.6131 --input imgs/adv --eval_adv --model insightface --thres 0.6351
```
### Purify genuie images
```
python defense.py --lambda_0 0.25 --sigma_y 0.15 --folder target --input imgs --eval_genuine --model insightface --thres 0.6351
```
### Adaptive attack
```
python attack.py --attack Adaptive --norm Linf --eps 0.03 --defense 0.25 0.15 --model insightface --thres 0.6351
```
where the following are partial options:
- `--model` allows `facenet` or `insightface`
- `--attack` allows `APGD` (white-box attack), `APGD_EOT` (adaptive attack), `Square` (black-box attack), or `Adaptive` (strong adaptive)'
- `--eval_genuine` runs the task that computes FRR for genuine images before or after purification
- `--eval_adv` runs the task that computes FAR and FRR for adv before or after purification

### Import for pre-processing
```
from fuctions.defense import iwmfdiff
```
```
def iwmfdiff(
	imgs_input: Tensor,
	lambda_0: float,
	sigma_y: float,
	s: int = 3,
	batch: int = 1,
	seed: int = None,
	) -> Tensor:
```

## Results

### Attack success rate / False acceptance rate (%)
* The security $FAR_{attack}$ is improved.

| Defense | $FAR_{SGADV}$ (seen) | $FAR_{FGSM}$ | $FAR_{PGD}$ | $FAR_{CW}$ | $FAR_{DI^2-FGSM}$ | $FAR_{TI-FGSM}$ | $FAR_{LGC}$ | $FAR_{BIM}$ |
|:---:|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Insightface | 100.0 | 100.0 | 100.0 | 100.0 | 95.00 | 93.17 | 93.73 | 91.50 |
| IWMF | 6.2 | 16.2 | 1.0 | 0.0 | 4.63 | 6.27 | 3.37 | 1.17 |
| IWMF-Diff | 3.2 | 15.6 | 0.8 | 0.2 | 28.53 | 33.00 | 23.97 | 10.87 |

### Authentication accuracy / False rejection rate (%)
* The authentication accuracy $FRR_{genuine}$ is preserved.
* The robustness of classifying adversarial examples as their true labels $FRR_{attack}$ is improved.

| Defense | $FRR_{genuine}$ | $FRR_{SGADV}$ (seen) | $FRR_{FGSM}$ | $FRR_{PGD}$ | $FRR_{CW}$ |
|:---:|:----:|:-----:|:-----:|:-----:|:-----:|
| Insightface | 0.28 | 98.30 | 6.34 | 51.92 | 42.12 |
| IWMF | 6.36 | 25.50 | 19.58 | 17.38 | 13.08 |
| IWMF-Diff | 3.22 | 12.06 | 8.28 | 9.22 | 6.18 |

## Citation
This work has been accepted by IEEE Transactions on Dependable and Secure Computing.

## Acknowledgement
The implementation is partially inspired by:
* [AutoAttack](https://github.com/fra31/auto-attack) (adversarial examples generation)
* [DDRM](https://github.com/bahjat-kawar/ddrm) (image restoration)

## Contacts
If you have any questions about our work, please do not hesitate to contact us by email.

Hanrui Wang: hanrui_wang@nii.ac.jp

