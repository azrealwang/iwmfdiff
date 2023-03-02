# Iterative Window Mean Filter: A New Image Filter Changes Diffusion-based Adversarial Purification

Hanrui Wang<sup>1</sup>, Shuo Wang<sup>2</sup>, Cunjian Chen<sup>1</sup>, Zhe Jin<sup>3</sup>, Soon Lay Ki<sup>1</sup>

<sup>1</sup> Monash University, <sup>2</sup> CSIRO, <sup>3</sup>Anhui Univeristy

Iterative window mean filter (IWMF) is a novel and super efficient non-deep-learning-based image filter, which gains comparable performance compared with the STOA adversarial defense. Taking advantage of IWMF, IWMF-Diff is a comprehensive framework for adversarial purification, which is applicable to any system as a pre-processing module against various attacks. IWMF-Diff gains superior performance than the SOTA defense.

<img src="figures/samples.jpg" alt="samples" style="width:400px;"/>

****
## Contents
* [Introduction](#Introduction)
* [Main requirements](#Main-requirements)
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
* Step 2: Further blur the image by Gaussian noise. This is essential for better restoration as DDRM is trained using Gaussian noise. Note that step 2 can be conducted using DDRM with step 3 together.
* Step 3: Restore image by DDRM. Robustness against both genuine images and adversarial examples raises.
* Step 4: Verify the pre-processed image by a regular authentication system. Note that users do not need to re-enroll due to the defense.

<img src="figures/pipeline.jpg" alt="pipeline" style="width:500px;"/>


## Main requirements

  * **torch == 1.1.0**
  * **torchvision == 0.3.0**
  * **tensorboardX == 1.7**
  * **bcolz == 1.2.1**
  * **Python 3**


## Installation
* git clone https://github.com/azrealwang/iwmfdiff.git
* pip3 install -r requirements.txt


## Data Preparation


## Pretrained Model

## Usage

## Results

## Citation
This work under review by [ACM CCS 2023](https://www.sigsac.org/ccs/CCS2023/).

## Acknowledgement
This implementation is based on / inspired by:
* [https://github.com/azrealwang/SGADV](https://github.com/azrealwang/SGADV) (generate adversarial examples)
* [https://github.com/bahjat-kawar/ddrm](https://github.com/bahjat-kawar/ddrm) (image restoration)


## Contacts
If you have any questions about our work, please do not hesitate to contact us by email.

Hanrui Wang: hanrui.wang@monash.edu

## Usage
```bash
# To train the model:
sh train.sh
# To evaluate the model:
(1)please first download the val data in https://github.com/ZhaoJ9014/face.evoLVe.PyTorch.
(2)set the checkpoint dir in config.py
sh evaluate.sh
```
You can change the experimental setting by simply modifying the parameter in the config.py

## Model
The IR101 pretrained model can be downloaded here. 
[Baidu Cloud](link: https://pan.baidu.com/s/1bu-uocgSyFHf5pOPShhTyA 
passwd: 5qa0), 
[Google Drive](https://drive.google.com/open?id=1upOyrPzZ5OI3p6WkA5D5JFYCeiZuaPcp)

## Result
The results of the released pretrained model are as follows:
|Data| LFW | CFP-FP | CPLFW | AGEDB | CALFW | IJBB (TPR@FAR=1e-4) | IJBC (TPR@FAR=1e-4) |
|:---:|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Result | 99.80 | 98.36 | 93.13 | 98.37 | 96.05 | 94.86 | 96.15 ||

The results are slightly different from the results in the paper because we replaced DataParallel with DistributedDataParallel and retrained the model.






# iwmfdiff

facenet
pytorch
eagerpy
numpy?
yaml

python main.py --dfr_model="insightface" --lambda_0=0 --sigma_y=-1 --batch_deno=10 --thresh=0.6131 --log_name="insightface-noDefense"

python main.py --dfr_model="insightface" --lambda_0=0.4 --sigma_y=-1 --batch_deno=10 --thresh=0.6611 --log_name="insightface-IWMF"

python main.py --dfr_model="insightface" --lambda_0=0.25 --sigma_y=0.15 --batch_deno=10 --thresh=0.6351 --log_name="insightface-IWMFDiff"