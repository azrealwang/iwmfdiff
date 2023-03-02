# iwmfdiff

facenet
pytorch
eagerpy
numpy?
yaml

python main.py --dfr_model="insightface" --lambda_0=0 --sigma_y=-1 --batch_deno=10 --thresh=0.6131 --log_name="insightface-noDefense"

python main.py --dfr_model="insightface" --lambda_0=0.4 --sigma_y=-1 --batch_deno=10 --thresh=0.6611 --log_name="insightface-IWMF"

python main.py --dfr_model="insightface" --lambda_0=0.25 --sigma_y=0.15 --batch_deno=10 --thresh=0.6351 --log_name="insightface-IWMFDiff"