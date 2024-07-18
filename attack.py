# This is a demo of defending Insightface
import codecs
import os
import argparse
import time
import torch

from math import ceil
from torch import Tensor
from functions.insightface.iresnet import iresnet100
from facenet_pytorch import InceptionResnetV1
from functions.utils import save_all_images, false_rate, FMR, load_samples, predict
from autoattack_fr.autopgd_base import APGDAttack
from autoattack_fr.square import SquareAttack


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', help='APGD (white-box attack), APGD_EOT (adaptive white-box attack), or Square (black-box attack)', type=str, default='APGD')
    parser.add_argument('--norm', help='Linf, L2, or L1', type=str, default='Linf')
    parser.add_argument('--eps', help='attack budget', type=float, default=0.03)
    parser.add_argument('--model', help='facenet or insightface', type=str, default='insightface')
    parser.add_argument('--thres', help='threshold', type=float, default=0.6131)
    parser.add_argument('--batch_size', help='batch size depends on memory', type=int, default=1)
    parser.add_argument('--input_target', help='target image path', type=str, default='imgs/target/lfw')
    parser.add_argument('--input_source', help='source image path', type=str, default='imgs/source/lfw')
    parser.add_argument('--output_adv', help='adv image path', type=str, default='imgs/adv')
    parser.add_argument('--output_log', help='log file path', type=str, default='logs/attack')
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
    model = args.model
    assert model in ['insightface', 'facenet']
    thres = args.thres
    batch_size = args.batch_size
    input_target = args.input_target
    input_source = args.input_source
    output_adv = args.output_adv
    output_log = args.output_log
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ## Start logging
    task_name = f'{attack_name}-{norm}-{eps}-{model}-{thres}'
    if not os.path.exists(output_log):
        os.makedirs(output_log)
    f = codecs.open(f'{output_log}/{task_name}.txt','a', 'utf-8')
    f.write(f'********************************START********************************\n')
    f.write(f'********Settings********\n\
    attack = {attack_name}\n\
    norm = {norm}\n\
    attack budget eps = {eps}\n\
    target model = {model}\n\
    threshold = {thres}\n')
    
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
    x_target, y_target = load_samples(input_target,end_idx,shape)
    x_target = Tensor(x_target)
    y_target = Tensor(y_target)
    target = predict(model,x_target,batch_size,device)
    x_source, y_source = load_samples(input_source,end_idx,shape)
    x_source = Tensor(x_source)
    y_source = Tensor(y_source)
    source = predict(model,x_source,batch_size,device)

    ## Compute clean accuracy
    far_a = FMR(source,target,thres) # attack success rate
    _, frr_adv = false_rate(source,y_source,source,y_source,thres) # adv as true labels
    print(f"Before attack: FAR_attack = {far_a}, FRR_adv = {frr_adv}")
    f.write(f"\n********PERFORMANCE BEFORE ATTACK********\n\
    FAR_attack = {far_a}\n\
    FRR_adv = {frr_adv}\n")
    
    ## Attack
    if attack_name == 'APGD':
        attack = APGDAttack(model,norm=norm,eps=eps,loss='fr_loss_targeted',device=device,thres=thres,n_iter=40)
    elif attack_name == 'APGD_EOT':
        attack = APGDAttack(model,norm=norm,eps=eps,loss='fr_loss_targeted',device=device,thres=thres,n_iter=40,eot_iter=20)
    elif attack_name == 'Square':
        attack = SquareAttack(model,norm=norm,eps=eps,device=device,thres=thres,n_queries=5000)
    
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
    f.write(f"Attack costs {end_time-start_time}s\n")
    # Save advs
    save_all_images(x_adv,y_source,f'{output_adv}/{task_name}')

    ## Compute attack accuracy
    x_adv, _ = load_samples(f'{output_adv}/{task_name}',end_idx,shape)
    x_adv = Tensor(x_adv)
    adv = predict(model,x_adv,batch_size,device)
    far_a = FMR(adv,target,thres) # attack success rate
    _, frr_adv = false_rate(adv,y_source,source,y_source,thres) # adv as true labels
    print(f"After attack: FAR_attack = {far_a}, FRR_adv = {frr_adv}")
    f.write(f"\n********PERFORMANCE AFTER ATTACK********\n\
    FAR_attack = {far_a}\n\
    FRR_adv = {frr_adv}\n")
    f.write(f"********************************END********************************\n\n")

    f.close()

if __name__ == "__main__":
    main()
