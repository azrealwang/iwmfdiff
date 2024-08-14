import argparse
import time
import torch
from math import ceil
from torch import Tensor
from functions.insightface.iresnet import iresnet100
from facenet_pytorch import InceptionResnetV1
from functions.utils import save_all_images, load_samples, predict
from evaluation import evaluate
from autoattack_fr.autopgd_base import APGDAttack
from autoattack_fr.square import SquareAttack


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    # Attack
    parser.add_argument('--attack', help='APGD (white-box attack), APGD_EOT (adaptive attack), Square (black-box attack), Adaptive (strong adaptive)', type=str, default='APGD')
    parser.add_argument('--norm', help='Linf, L2, or L1', type=str, default='Linf')
    parser.add_argument('--eps', help='attack budget', type=float, default=0.03)
    parser.add_argument('--seed', help='seed', type=int, default=None)
    parser.add_argument('--defense', help='Defense setting only for Adaptive', type=int, nargs=2, default=None)
    # Model and Input
    parser.add_argument('--model', help='facenet or insightface', type=str, required=True)
    parser.add_argument('--thres', help='threshold', type=float, required=True)
    parser.add_argument('--batch_size', help='batch size depends on memory', type=int, default=1)
    parser.add_argument('--input_target', help='target image path', type=str, default='imgs/target/')
    parser.add_argument('--input_source', help='source image path', type=str, default='imgs/source/')
    # Output
    parser.add_argument('--output_adv', help='adv image path', type=str, default='imgs/adv')

    args = parser.parse_args()

    return args

def main() -> None:
    ## Settings
    args = parse_args_and_config()
    attack_name = args.attack
    assert attack_name in ['APGD', 'APGD_EOT', 'Square', 'Adaptive']
    norm = args.norm
    assert norm in ['Linf', 'L2', 'L1']
    eps = args.eps
    seed = args.seed
    defense = args.defense
    if attack_name == 'Adaptive' and defense is None:
        raise ValueError('Adaptive attack requires defense settings')
    model = args.model
    assert model in ['insightface', 'facenet']
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
    if model == 'insightface':
        model = iresnet100(pretrained=True).eval().to(device)
        shape = (112,112)
    elif model == 'facenet':
        model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        shape = (160,160)
    else:
        raise ValueError("unsupported model")
    
    ## Load inputs
    x_target, y_target = load_samples(input_target,shape=shape)
    x_target = Tensor(x_target)
    y_target = Tensor(y_target)
    target = predict(model,x_target,batch_size,device)
    x_source, y_source = load_samples(input_source,shape=shape)
    x_source = Tensor(x_source)
    y_source = Tensor(y_source)
    
    ## Attack
    if attack_name == 'APGD':
        attack = APGDAttack(model,norm=norm,eps=eps,loss='fr_loss_targeted',device=device,thres=thres,n_iter=40,seed=seed)
    elif attack_name == 'APGD_EOT':
        attack = APGDAttack(model,norm=norm,eps=eps,loss='fr_loss_targeted',device=device,thres=thres,n_iter=40,eot_iter=20,seed=seed)
    elif attack_name == 'Square':
        attack = SquareAttack(model,norm=norm,eps=eps,device=device,thres=1,n_queries=20000,seed=seed)
    elif attack_name == 'Adaptive':
        attack = APGDAttack(model,norm=norm,eps=eps,loss='fr_loss_targeted',device=device,thres=thres,n_iter=40,seed=seed,adaptive=True,defense=defense)

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
