import argparse
import torch
from torch import Tensor
from iwmfdiff.insightface.iresnet import iresnet100
from facenet_pytorch import InceptionResnetV1
from iwmfdiff.utils import false_rate, FMR, load_samples, predict


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    # Task
    parser.add_argument('--eval_genuine', help='compute FRR for genuine before or after purification', action='store_true')
    parser.add_argument('--eval_adv', help='compute FAR and FRR for adv before or after purification', action='store_true')
    # Data
    parser.add_argument('--input_eval', help='test image path', type=str, required=True)
    parser.add_argument('--input_target', help='target image path', type=str, default='imgs/target/')
    parser.add_argument('--input_source', help='source image path', type=str, default='imgs/source/')
    # Model
    parser.add_argument('--model', help='facenet or insightface', type=str, required=True)
    parser.add_argument('--thres', help='threshold', type=float, required=True)
    parser.add_argument('--batch_size', help='batch size depends on memory', type=int, default=1)
    
    args = parser.parse_args()

    return args

def evaluate(args) -> None:
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
    assert model in ['insightface', 'facenet']
    thres = args.thres
    batch_size = args.batch_size
    input_eval = args.input_eval
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    x_test, y_test = load_samples(input_eval,shape=shape)
    x_test = Tensor(x_test)
    y_test = Tensor(y_test)
    test = predict(model,x_test,batch_size,device)
    x_target, y_target = load_samples(input_target,shape=shape)
    x_target = Tensor(x_target)
    y_target = Tensor(y_target)
    target = predict(model,x_target,batch_size,device)
    if args.eval_adv:
        x_source, y_source = load_samples(input_source,shape=shape)
        x_source = Tensor(x_source)
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