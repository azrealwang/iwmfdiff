################################
#!/bin/sh
#SBATCH --job-name=apgd_linf_0.03_if
#SBATCH --out="logs/apgd_linf_0.03_if.txt"
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla_a100:1

python -u attack.py --attack APGD --eps 0.015 --model insightface --thres 0.6131 --batch_size 100
python -u attack.py --attack APGD --eps 0.03 --model insightface --thres 0.6131 --batch_size 100
python -u attack.py --attack APGD --eps 0.06 --model insightface --thres 0.6131 --batch_size 100
python -u attack.py --attack APGD --eps 1 --model insightface --thres 0.6131 --batch_size 100
python -u attack.py --attack APGD --eps 0.03 --model facenet --thres 0.7056 --batch_size 100

python -u attack.py --attack APGD_EOT --eps 0.03 --model insightface --thres 0.6131 --batch_size 100

python -u attack.py --attack Square --eps 0.015 --model insightface --thres 0.6131 --batch_size 100
python -u attack.py --attack Square --eps 0.03 --model insightface --thres 0.6131 --batch_size 100
python -u attack.py --attack Square --eps 0.06 --model insightface --thres 0.6131 --batch_size 100
python -u attack.py --attack Square --eps 1 --model insightface --thres 0.6131 --batch_size 100
python -u attack.py --attack Square --eps 0.03 --model facenet --thres 0.7056 --batch_size 100

#SBATCH --nodelist=gpuhost04

scontrol show job 559928