#!/bin/sh
#SBATCH --job-name=eval12
#SBATCH --out="logs/eval12.txt"
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla_a100:1

python -u evaluation.py --input_eval imgs/purified/achive/Square-Linf-1.0-insightface-0.6131-0.25-0.15-3 --eval_adv --model insightface --thres 0.6131 --batch_size 100
