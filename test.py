from functions.utils import save_all_images, load_samples
from torch import Tensor

images, labels = load_samples('imgs/target/lfw',500)
images_new = images.copy()
labels_new = labels.copy()
images_new[0:250] = images[250:]
images_new[250:] = images[0:250]
labels_new[0:250] = labels[250:]
labels_new[250:] = labels[0:250]
save_all_images(Tensor(images_new),labels_new,'imgs/source/lfw')