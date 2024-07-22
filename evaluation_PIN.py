import argparse
import torch
from torch import Tensor
import torchvision.transforms as transforms
from functions.insightface.iresnet import iresnet100
from facenet_pytorch import InceptionResnetV1
from PIN.config_test import Config
from PIN.defense_model import DefenseModel
from functions.utils import false_rate, FMR, load_samples, predict


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_genuine', help='evaluate genuine before and after purification', action='store_true')
    parser.add_argument('--eval_adv', help='evaluate adv before and after purification', action='store_true')
    parser.add_argument('--input_target', help='target image path', type=str, default='imgs/target/lfw')
    parser.add_argument('--input_source', help='source image path', type=str, default='imgs/source/lfw')
    parser.add_argument('--model', help='PIN', type=str, default='PIN')
    parser.add_argument('--thres', help='threshold', type=float, default=0.6131)
    parser.add_argument('--batch_size', help='batch size depends on memory', type=int, default=1)
    parser.add_argument('--input_eval', help='input image path', type=str, required=True)
    
    args = parser.parse_args()

    return args

def evaluate(args) -> None:
    end_idx = 500
    ## Checking parameters
    if not args.eval_genuine and not args.eval_adv:
        return 
    if args.eval_genuine and args.eval_adv:
        raise ValueError('Cannot evaluate genuine and adv images together')
    if args.input_eval is None:
        raise ValueError('Must have an input')
    ## Settings
    input_target = args.input_target
    input_source = args.input_source
    model = args.model
    assert model in ['PIN']
    thres = args.thres
    batch_size = args.batch_size
    input_eval = args.input_eval
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    x_test, y_test = load_samples(input_eval,end_idx,shape)
    x_test = trans(Tensor(x_test))
    y_test = Tensor(y_test)
    test = predict(model,x_test,batch_size,device)
    x_target, y_target = load_samples(input_target,end_idx,shape)
    x_target = trans(Tensor(x_target))
    y_target = Tensor(y_target)
    target = predict(model,x_target,batch_size,device)
    if args.eval_adv:
        x_source, y_source = load_samples(input_source,end_idx,shape)
        x_source = trans(Tensor(x_source))
        y_source = Tensor(y_source)
        source = predict(model,x_source,batch_size,device)

    ## Compute accuracy
    if args.eval_genuine:
        _, frr_g = false_rate(test,y_test,target,y_target,thres) # genuine accuracy
        print(f"FRR_genuine = {frr_g}")
    elif args.eval_adv:
        fmr = FMR(test,target,thres) # attack success rate
        _, frr_adv = false_rate(test,y_test,source,y_source,thres) # adv as true labels
        print(f"FAR_attack = {fmr}, FRR_adv = {frr_adv}")

if __name__ == "__main__":
    args = parse_args_and_config()
    evaluate(args)