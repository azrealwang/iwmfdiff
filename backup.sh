################################
#!/bin/sh
#SBATCH --job-name=apgd_0.03_fn_adaptive_diffpure
#SBATCH --out="logs/apgd_0.03_fn_adaptive_diffpure.txt"
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla_a100:1

python -u attack.py --attack APGD --eps 0.015 --model insightface --thres 0.6131 --batch_size 100
python -u attack.py --attack APGD --eps 0.03 --model insightface --thres 0.6131 --batch_size 100
python -u attack.py --attack APGD --eps 0.06 --model insightface --thres 0.6131 --batch_size 100
python -u attack.py --attack APGD --eps 1 --model insightface --thres 0.6131 --batch_size 100
python -u attack.py --attack APGD --eps 0.03 --model facenet --thres 0.7056 --batch_size 100

python -u attack.py --attack APGD_EOT --eps 0.03 --model insightface --thres 0.6131 --batch_size 100
python -u attack.py --attack APGD_EOT --eps 0.03 --model facenet --thres 0.7056 --batch_size 100

python -u attack.py --attack Square --eps 0.015 --model insightface --thres 0.6131 --batch_size 100
python -u attack.py --attack Square --eps 0.03 --model insightface --thres 0.6131 --batch_size 100
python -u attack.py --attack Square --eps 0.06 --model insightface --thres 0.6131 --batch_size 100
python -u attack.py --attack Square --eps 1 --model insightface --thres 0.6131 --batch_size 100
python -u attack.py --attack Square --eps 0.03 --model facenet --thres 0.7056 --batch_size 100

python -u defense.py --folder APGD-Linf-0.03-insightface-0.6131 --input imgs/adv --lambda_0 0 --sigma_y 0.15 --batch_size 50 --eval_adv --model insightface --thres 0.7119

python -u evaluation.py --input_eval imgs/purified/achive/APGD-Linf-0.015-insightface-0.6131-0.0-0.15-3 --eval_adv --model insightface --thres 0.6131 --batch_size 100
python -u evaluation_PIN.py --input_eval imgs/adv/achive/APGD-Linf-0.03-insightface-0.6131 --eval_adv --model PIN --thres 0.5890 --batch_size 100

#SBATCH --nodelist=gpuhost04

scontrol show job 559928