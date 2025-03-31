import torch
from iwmfdiff.defense import ddrm

img = torch.randn(1, 3, 256, 256)  # Dummy input
result = ddrm(img, data="celeba_hq")
print("Test completed successfully!")