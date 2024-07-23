import argparse
import time
import torch
from math import ceil
from torch import Tensor
import torchvision.transforms as transforms
from functions.insightface.iresnet import iresnet100
from facenet_pytorch import InceptionResnetV1
from PIN.config_test import Config
from PIN.defense_model import DefenseModel
from functions.utils import save_all_images, load_samples, predict
from evaluation_PIN import evaluate
from autoattack_fr.autopgd_base import APGDAttack
from autoattack_fr.square import SquareAttack


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', help='APGD (white-box attack), APGD_EOT (adaptive white-box attack), or Square (black-box attack)', type=str, default='APGD')
    parser.add_argument('--norm', help='Linf, L2, or L1', type=str, default='Linf')
    parser.add_argument('--eps', help='attack budget', type=float, default=0.03)
    parser.add_argument('--seed', help='seed', type=int, default=None)
    parser.add_argument('--model', help='PIN', type=str, default='PIN')
    parser.add_argument('--thres', help='threshold', type=float, default=0.6131)
    parser.add_argument('--batch_size', help='batch size depends on memory', type=int, default=1)
    parser.add_argument('--input_target', help='target image path', type=str, default='imgs/target/lfw')
    parser.add_argument('--input_source', help='source image path', type=str, default='imgs/source/lfw')
    parser.add_argument('--output_adv', help='adv image path', type=str, default='imgs/adv')
    args = parser.parse_args()

    return args

def main() -> None:
    end_idx = 500
    ## Settings
    args = parse_args_and_config()
    attack_name = args.attack
    assert attack_name in ['APGD', 'APGD_EOT', 'Square']
    norm = args.norm
    assert norm in ['Linf', 'L2', 'L1']
    eps = args.eps
    seed = args.seed
    model = args.model
    assert model in ['PIN']
    thres = args.thres
    batch_size = args.batch_size
    input_target = args.input_target
    input_source = args.input_source
    output_adv = args.output_adv

    args.eval_adv = True
    args.eval_genuine = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    folder = f'{attack_name}-{norm}-{eps}-{model}-{thres}'
    
    ## Compute clean accuracy
    print(f"Before attack......")
    args.input_eval = args.input_source
    evaluate(args)

    ## Load Model
    if model == 'PIN':
        trans = transforms.Compose([
            transforms.Grayscale(1),
        ])
        config = Config()
        model = DefenseModel(config)
        shape = (112,112)
    else:
        raise ValueError("unsupported model")
    
    ## Load inputs
    x_target, y_target = load_samples(input_target,end_idx,shape)
    x_target = trans(Tensor(x_target))
    y_target = Tensor(y_target)
    target = predict(model,x_target,batch_size,device)
    x_source, y_source = load_samples(input_source,end_idx,shape)
    x_source = trans(Tensor(x_source))
    y_source = Tensor(y_source)
    source = predict(model,x_source,batch_size,device)
    
    ## Attack
    if attack_name == 'APGD':
        attack = APGDAttack(model,norm=norm,eps=eps,loss='fr_loss_targeted',device=device,thres=thres,n_iter=40,seed=seed)
    elif attack_name == 'APGD_EOT':
        attack = APGDAttack(model,norm=norm,eps=eps,loss='fr_loss_targeted',device=device,thres=thres,n_iter=40,eot_iter=20,seed=seed)
    elif attack_name == 'Square':
        attack = SquareAttack(model,norm=norm,eps=eps,device=device,thres=1,n_queries=20000,seed=seed)
    
    x_adv = Tensor([])
    start_time = time.time()
    for i in range(ceil(len(y_target)/batch_size)):
        print(f"Batch: {i + 1}")
        start = i * batch_size
        if i == ceil(len(y_target)/batch_size) - 1:
            batch_size = len(y_target) - batch_size * i
        x_adv_batch = attack.perturb(x_source[start:start+batch_size],target[start:start+batch_size]).cpu()
        x_adv = torch.cat((x_adv, x_adv_batch), 0)
    end_time = time.time()
    print(f"Attack costs {end_time-start_time}s")
    # Save advs
    save_all_images(x_adv,y_source,f'{output_adv}/{folder}')

    ## Compute attack accuracy
    print(f"After attack......")
    args.input_eval = f'{output_adv}/{folder}'
    evaluate(args)

if __name__ == "__main__":
    main()
