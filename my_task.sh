#!/bin/sh
#SBATCH --job-name=square_0.03_fn
#SBATCH --out="logs/square_0.03_fn.txt"
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla_a100:1

python -u attack.py --attack Square --eps 0.03 --model facenet --thres 0.7056 --batch_size 100
