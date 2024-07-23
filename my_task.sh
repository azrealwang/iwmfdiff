#!/bin/sh
#SBATCH --job-name=diffpure_apgd_0.03_fn_adaptive_diffpure
#SBATCH --out="logs/40/adaptive/diffpure_apgd_0.03_fn_adaptive_diffpure.txt"
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla_a100:1

python -u defense.py --folder APGD-Linf-0.03-facenet-0.7407-diffpure --input imgs/adv/40/adaptive --output imgs/purified/40/adaptive --lambda_0 0 --sigma_y 0.15 --batch_size 50 --eval_adv --model facenet --thres 0.7407 --seed 0