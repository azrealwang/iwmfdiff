import codecs
import os
import argparse
from facenet_pytorch import InceptionResnetV1
from insightface.iresnet import iresnet100
from iwmfdiff.models import PyTorchModel
from iwmfdiff.utils import samples, save_all_images, iwmf, ddrm, false_rate, FMR


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', help='how many images from each subject', type=int, default=10)
    parser.add_argument('--subject', help='how many subjects', type=int, default=2)
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
    sample = args.sample
    subject = args.subject
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
    sample = {sample}\n\
    subject = {subject}\n\
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
    total = sample * subject
    images, labels = samples(fmodel,dataset=f"inputs/genuine",batchsize=total,shape=shape)
    target_features = fmodel(images)
    images, _ = samples(fmodel,dataset=f"inputs/source",batchsize=total,shape=shape)
    source_features = fmodel(images)
    del images
    
    ## Genuine
    images, _ = samples(fmodel,dataset=f"inputs/genuine",batchsize=total,shape=shape)
    if lambda_0>0 or sigma_y>0:
        images = iwmf(images,lambda_0,s) # IWMF
        path_genuine_blur = os.path.join(outputs_path, 'genuine_blur')
        if not os.path.exists(path_genuine_blur):
            os.makedirs(path_genuine_blur)
        save_all_images(images,subject,sample,path_genuine_blur)
    if sigma_y>0:
        images, _ = samples(fmodel,dataset=path_genuine_blur,batchsize=total,shape=(256,256)) # required shape by DDRM
        print("**********************denoising genuine images...******************************")
        images = ddrm(images,sigma_0=sigma_y,batch=batch_deno) # IWMF-Diff
        path_genuine_deno = os.path.join(outputs_path, 'genuine_deno')
        if not os.path.exists(path_genuine_deno):
            os.makedirs(path_genuine_deno)
        save_all_images(images,subject,sample,path_genuine_deno)
        images, _ = samples(fmodel,dataset=path_genuine_deno,batchsize=total,shape=shape)   
    genuine_features = fmodel(images)
    
    ## Adv
    images, _ = samples(fmodel,dataset=f"inputs/adv",batchsize=total,shape=shape)
    if lambda_0>0 or sigma_y>0:
        images = iwmf(images,lambda_0,s) # IWMF
        path_adv_blur = os.path.join(outputs_path, 'adv_blur')
        if not os.path.exists(path_adv_blur):
            os.makedirs(path_adv_blur)
        save_all_images(images,subject,sample,path_adv_blur)
    if sigma_y>0:
        images, _ = samples(fmodel,dataset=path_adv_blur,batchsize=total,shape=(256,256)) # required shape by DDRM
        print("**********************denoising adv images...******************************")
        images = ddrm(images,sigma_0=sigma_y,batch=batch_deno) # IWMF-Diff
        path_adv_deno = os.path.join(outputs_path, 'adv_deno')
        if not os.path.exists(path_adv_deno):
            os.makedirs(path_adv_deno)
        save_all_images(images,subject,sample,path_adv_deno)
        images, _ = samples(fmodel,dataset=path_adv_deno,batchsize=total,shape=shape)   
    adv_features = fmodel(images)
    del images, fmodel
    
    ## Score
    far_attack, _ = FMR(adv_features,target_features,threshold,sample) # attack success rate
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
