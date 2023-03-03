import codecs
import os
import argparse
from facenet_pytorch import InceptionResnetV1
from functions.insightface.iresnet import iresnet100
from functions.models import PyTorchModel
from functions.utils import samples, save_all_images, false_rate, FMR
from functions.defense import iwmfdiff


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dfr_model', help='deep learning models', type=str, default='insightface',choices=['insightface', 'facenet'])
    parser.add_argument('--lambda_0', help='window amount >0; 0 indicates no blurring', type=float, default=0.25)
    parser.add_argument('--sigma_y', help='Gaussian standard deviation in [0,1]; -1 indicates no denoising', type=float, default=0.15)
    parser.add_argument('--s', help='window size (px)', type=int, default=3)
    parser.add_argument('--batch_deno', help='batch size for ddrm processing; depend on memory', type=int, default=1)
    parser.add_argument('--thresh', help='threshold', type=float, default=0.6351)
    parser.add_argument('--log_name', help='log file name', type=str, default='IWMFDiff')
    parser.add_argument('--logs_path', help='log file path', type=str, default='logs')
    parser.add_argument('--outputs_path', help='output images path', type=str, default='outputs')
    args = parser.parse_args()

    return args

def main() -> None:
    ## Settings
    args = parse_args_and_config()
    dfr_model = args.dfr_model
    threshold = args.thresh
    lambda_0 = args.lambda_0
    s = args.s
    sigma_y = args.sigma_y
    batch_deno = args.batch_deno
    log_name = args.log_name
    logs_path = args.logs_path
    outputs_path = args.outputs_path
    del args
    
    ## Start logging
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    f = codecs.open(f'{logs_path}/{log_name}.txt','a', 'utf-8')
    f.write(f'********************************START********************************\n')
    f.write(f'********Settings********\n\
    deep face model = {dfr_model}\n\
    threshold = {threshold}\n\
    window amount lambda = {lambda_0}\n\
    window size s = {s}px\n\
    Gaussian standard deviation sigma_y = {sigma_y}\n')
    
    ## Load Model
    if dfr_model == 'insightface':
        model = iresnet100(pretrained=True).eval()
        shape = (112,112) # resize images
    elif dfr_model == 'facenet':
        model = InceptionResnetV1(pretrained='vggface2').eval()
        shape = (160,160) # resize images
    else:
        raise ValueError("unsupported model")
    mean = [0.5] * 3
    std = [0.5] * 3
    preprocessing = dict(mean=mean,std=std,axis=-3)
    bounds = (0, 1)
    fmodel = PyTorchModel(model,bounds=bounds,preprocessing=preprocessing)
    
    ## Load clean features
    total = len(os.listdir(f'inputs/genuine'))
    images, labels = samples(fmodel,dataset=f"inputs/genuine",batchsize=total,shape=shape)
    target_features = fmodel(images)
    images, _ = samples(fmodel,dataset=f"inputs/source",batchsize=total,shape=shape)
    source_features = fmodel(images)
    del images
    
    ## Genuine
    images, _ = samples(fmodel,dataset=f"inputs/genuine",batchsize=total,shape=shape)
    if lambda_0>0 or sigma_y>0:
        images = iwmfdiff(images,lambda_0,sigma_y,s,batch_deno) # IWMFDiff
        path_genuine_blur = os.path.join(outputs_path, 'genuine')
        if not os.path.exists(path_genuine_blur):
            os.makedirs(path_genuine_blur)
        save_all_images(images,labels,path_genuine_blur)
    genuine_features = fmodel(images)
    
    ## Adv
    images, _ = samples(fmodel,dataset=f"inputs/adv",batchsize=total,shape=shape)
    if lambda_0>0 or sigma_y>0:
        images = iwmfdiff(images,lambda_0,sigma_y,s,batch_deno) # IWMFDiff
        path_adv_blur = os.path.join(outputs_path, 'adv')
        if not os.path.exists(path_adv_blur):
            os.makedirs(path_adv_blur)
        save_all_images(images,labels,path_adv_blur)
    adv_features = fmodel(images)
    del images, fmodel
    
    ## Score
    far_attack = FMR(adv_features,target_features,threshold) # attack success rate
    _, frr = false_rate(genuine_features,labels,target_features,labels,threshold) # genuine accuracy
    _, frr_adv = false_rate(adv_features,labels,source_features,labels,threshold) # adv as true labels
    print(f"FAR_attack = {far_attack}, FRR_genuine = {frr}, FRR_adv = {frr_adv}")
    f.write(f"\n********PERFORMANCE********\n\
    FAR_attack = {far_attack}\n\
    FRR_genuine = {frr}\n\
    FRR_adv = {frr_adv}\n")
    f.write(f"********************************END********************************\n\n")

    f.close()

if __name__ == "__main__":
    main()
