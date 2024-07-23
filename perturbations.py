import os
import numpy as np
from torch import Tensor
from functions.utils import load_samples,save_all_images

def rgb2gray(rgb):
    r, g, b = rgb[0], rgb[1], rgb[2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# Settings
orig_path = f'imgs/source/lfw'
test_path = f'imgs/purified/archive'
folder_name = 'APGD-Linf-0.06-insightface-0.6131-0.25-0.15-3'
outputs_path = f"imgs/tmp/"
idx = 0
# model: 20,  (91), 100, (210), 223, 229, 233, 281, 328, 358, 377,402, 446, 587, 668, 720, 728, 766, 841;
# reverse: 22,  32, (118), 237, 266, 368, (437), 597, 655, (695), 734, 900, 995
# Greedy: Cifar-10 548, (735)
# Greedy: Imagenet 0

# Load images
path = os.path.join(test_path, folder_name)
x_test, _ = load_samples(path,idx+1)
x_test = x_test[-1]
shape = x_test.shape[1:]
x_orig, y = load_samples(orig_path,idx+1,shape)
x_orig = x_orig[-1]
y = y[-1]


# Generate perturbations
eps = np.max(np.abs(x_test-x_orig))
print(eps*255)
if eps == 0:
    eps = 0.00000001
#diff = ((x_test-x_orig)+eps/255)/2*255/eps # normalize
diff = (x_test-x_orig)*255 # normalize
diff = rgb2gray(diff) # rgb to gray
if not os.path.exists(outputs_path):
    os.makedirs(outputs_path)
save_all_images(Tensor(np.array([diff])),[y],outputs_path)