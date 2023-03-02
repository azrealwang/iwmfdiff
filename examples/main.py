import codecs
import os
from facenet_pytorch import InceptionResnetV1
from insightface.iresnet import iresnet100
from iwmfdiff.models import PyTorchModel
from iwmfdiff.utils import samples, iwmf, save_all_images, ddrm, false_rate, FMR

def main() -> None:
    ## Settings
    sample = int(10) # how many images from each subject
    subject = int(2) # how many subjects
    dfr_model = 'insightface'  # facenet, insightface
    lambda_0 = 0.25 # window amount in [0,1]
    sigma_y = 0.15 # Gaussian standard deviation in [0,1]; -1 indicates IWMF only
    s = 3 # window size (px)
    batch_deno = 10 # depend on GPU memory
    class thresh:
        clean = 0.6131 # threshold without defense
        iwmf = 0.6611 # threshold with IWMF
        iwmfdiff = 0.6351 # threshold with IWMF-Diff
    log_name = 'IWMFDiff'
    logs_path = 'logs'
    outputs_path = 'outputs'
    
    ## Start logging
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    f = codecs.open(f'{logs_path}/{log_name}.txt','a', 'utf-8')
    f.write(f"sample = {sample},\n\
        subject = {subject},\n\
        deep face model = {dfr_model},\n\
        window amount lambda = {lambda_0},\n\
        window size s = {s}px,\n\
        Gaussian standard deviation sigma_y = {sigma_y},\n\
        \n\
        ********PERFORMANCE********\n")
    
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
    
    ## Without defense
    # Genuine
    images, _ = samples(fmodel,dataset=f"inputs/genuine",batchsize=total,shape=shape)
    genuine_features = fmodel(images)
    # Adv
    images, _ = samples(fmodel,dataset=f"inputs/adv",batchsize=total,shape=shape)
    adv_features = fmodel(images)
    del images
    # Score
    threshold = thresh.clean
    far_attack, _ = FMR(adv_features,target_features,threshold,sample) # attack success rate
    _, frr = false_rate(genuine_features,labels,target_features,labels,threshold) # genuine accuracy
    _, frr_adv = false_rate(adv_features,labels,source_features,labels,threshold) # adv as true labels
    print(f"without defense: threshold = {threshold}, FAR_attack = {far_attack}, FRR_genuine = {frr}, FRR_adv = {frr_adv}")
    f.write(f"without defense: threshold = {threshold}, FAR_attack = {far_attack}, FRR_genuine = {frr}, FRR_adv = {frr_adv}\n")
    del genuine_features, adv_features, threshold, far_attack, frr, frr_adv
    
    ## IWMF
    # Genuine
    images, _ = samples(fmodel,dataset=f"inputs/genuine",batchsize=total,shape=shape)
    processed_images = iwmf(images,lambda_0,s) # IWMF
    path_genuine_blur = os.path.join(outputs_path, 'genuine_blur')
    if not os.path.exists(path_genuine_blur):
        os.makedirs(path_genuine_blur)
    save_all_images(processed_images,subject,sample,path_genuine_blur)
    genuine_features = fmodel(processed_images)
    # Adv
    images, _ = samples(fmodel,dataset=f"inputs/adv",batchsize=total,shape=shape)
    processed_images = iwmf(images,lambda_0,s) # IWMF
    path_adv_blur = os.path.join(outputs_path, 'adv_blur')
    if not os.path.exists(path_adv_blur):
        os.makedirs(path_adv_blur)
    save_all_images(processed_images,subject,sample,path_adv_blur)
    adv_features = fmodel(processed_images)
    del images, processed_images
    if sigma_y<0: # Score for IWMF
        threshold = thresh.iwmf
        far_attack, _ = FMR(adv_features,target_features,threshold,sample) # attack success rate
        _, frr = false_rate(genuine_features,labels,target_features,labels,threshold) # genuine accuracy
        _, frr_adv = false_rate(adv_features,labels,source_features,labels,threshold) # adv as true labels
        print(f"with IWMF: threshold = {threshold}, FAR_attack = {far_attack}, FRR_genuine = {frr}, FRR_adv = {frr_adv}")
        f.write(f"with IWMF: threshold = {threshold}, FAR_attack = {far_attack}, FRR_genuine = {frr}, FRR_adv = {frr_adv}\n")
        del genuine_features, adv_features, threshold, far_attack, frr, frr_adv
        
    else: ## IWMF-Diff
        del genuine_features, adv_features
        # Genuine
        images, _ = samples(fmodel,dataset=path_genuine_blur,batchsize=total,shape=(256,256)) # required shape by DDRM
        print("**********************denoising genuine images...******************************")
        processed_images = ddrm(images,sigma_0=sigma_y,batch=batch_deno) # denoising
        path_genuine_deno = os.path.join(outputs_path, 'genuine_deno')
        if not os.path.exists(path_genuine_deno):
            os.makedirs(path_genuine_deno)
        save_all_images(processed_images,subject,sample,path_genuine_deno)
        images, _ = samples(fmodel,dataset=path_genuine_deno,batchsize=total,shape=shape)
        genuine_features = fmodel(images)
        # Adv
        images, _ = samples(fmodel,dataset=path_adv_blur,batchsize=total,shape=(256,256)) # required shape by DDRM
        print("**********************denoising adv images...******************************")
        processed_images = ddrm(images,sigma_0=sigma_y,batch=batch_deno) # denoising
        path_adv_deno = os.path.join(outputs_path, 'adv_deno')
        if not os.path.exists(path_adv_deno):
            os.makedirs(path_adv_deno)
        save_all_images(processed_images,subject,sample,path_adv_deno)
        images, _ = samples(fmodel,dataset=path_adv_deno,batchsize=total,shape=shape)
        adv_features = fmodel(images)
        del images, processed_images
        # Score
        threshold = thresh.iwmfdiff
        far_attack, _ = FMR(adv_features,target_features,threshold,sample) # attack success rate
        _, frr = false_rate(genuine_features,labels,target_features,labels,threshold) # genuine accuracy
        _, frr_adv = false_rate(adv_features,labels,source_features,labels,threshold) # adv as true labels
        print(f"with IWMF-Diff: threshold = {threshold}, FAR_attack = {far_attack}, FRR_genuine = {frr}, FRR_adv = {frr_adv}")
        f.write(f"with IWMF-Diff: threshold = {threshold}, FAR_attack = {far_attack}, FRR_genuine = {frr}, FRR_adv = {frr_adv}\n")
        del genuine_features, adv_features, threshold, far_attack, frr, frr_adv

    f.close()

if __name__ == "__main__":
    main()
