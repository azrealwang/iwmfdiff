#!/bin/sh
#SBATCH --job-name=apgd_0.03_if_adaptive_diffpure
#SBATCH --out="logs/apgd_0.03_if_adaptive_diffpure.txt"
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla_a100:1

python -u attack.py --attack APGD --eps 0.03 --model insightface --thres 0.7119 --batch_size 50
