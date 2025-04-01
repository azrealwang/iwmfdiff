import argparse
import torch
from torch import Tensor
from iwmfdiff.utils import save_all_images, load_samples
from iwmfdiff.defense import iwmfdiff
from .evaluation import evaluate


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    # Defense
    parser.add_argument('--lambda_0', help='window amount >= 0; 0 indicates DiffPure', type=float, default=0.25)
    parser.add_argument('--sigma_y', help='Gaussian standard deviation in [0,1]; 0 indicates IWMF', type=float, default=0.15)
    parser.add_argument('--s', help='window size (px) of IWMF', type=int, default=3)
    parser.add_argument('--seed', help='seed', type=int, default=None)
    parser.add_argument('--batch_size', help='batch size depends on memory', type=int, default=1)
    # Input and Output
    parser.add_argument('--folder', help='input folder name', type=str, required=True)
    parser.add_argument('--input', help='input folder path, excluding folder name', type=str, required=True)
    parser.add_argument('--input_target', help='target image path', type=str, default='imgs/target/')
    parser.add_argument('--input_source', help='source image path', type=str, default='imgs/source/')
    parser.add_argument('--output', help='purified image path', type=str, default='imgs/purified')
    # evaluation
    parser.add_argument('--eval_genuine', help='compute FRR for genuine before or after purification', action='store_true')
    parser.add_argument('--eval_adv', help='compute FAR and FRR for adv before or after purification', action='store_true')
    parser.add_argument('--model', help='facenet or insightface', type=str, required=True)
    parser.add_argument('--thres', help='threshold', type=float, required=True)

    args = parser.parse_args()

    return args

def main() -> None:
    ## Settings
    args = parse_args_and_config()
    if args.lambda_0 == 0 and args.sigma_y == 0:
        raise ValueError('both lambda_0 and sigma_y are 0')
    lambda_0 = args.lambda_0
    assert lambda_0 >= 0
    sigma_y = args.sigma_y
    assert 0 <= sigma_y <= 1
    s = args.s
    assert s > 0
    seed = args.seed
    batch_size = args.batch_size
    folder = args.folder
    input = args.input
    output = args.output

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    folder_p = f'{folder}-{lambda_0}-{sigma_y}-{s}'

    ## Compute clean accuracy
    print(f"Before purification......")
    args.input_eval = f'{input}/{folder}'
    evaluate(args)
    
    ## Purification
    x, y = load_samples(f'{input}/{folder}')
    x = Tensor(x).to(device)
    y = Tensor(y).to(device)
    x_p = iwmfdiff(x,lambda_0,sigma_y,s,batch_size,seed) # IWMFDiff
    save_all_images(x_p,y,f'{output}/{folder_p}')
    
    ## Compute purification accuracy
    print(f"After purification......")
    args.input_eval = f'{output}/{folder_p}'
    evaluate(args)

if __name__ == "__main__":
    main()
