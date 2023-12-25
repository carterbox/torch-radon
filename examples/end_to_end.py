import torch
import torch.nn as nn
import numpy as np
from torch_radon import ParallelBeam as Radon

batch_size = 8
n_angles = 64
image_size = 128
channels = 4
det_count = image_size

device = torch.device('cuda')
criterion = nn.L1Loss()

# Instantiate a model for the sinogram and one for the image
sino_model = nn.Conv2d(1, channels, 5, padding=2).to(device)
image_model = nn.Conv2d(channels, 1, 3, padding=1).to(device)

# create empty images
x = torch.FloatTensor(batch_size, 1, image_size, image_size).to(device)

# instantiate Radon transform
angles = np.linspace(0, np.pi, n_angles)
radon = Radon(det_count, angles)

# forward projection
sinogram = radon.forward(x)

# apply sino_model to sinograms
filtered_sinogram = sino_model(sinogram)

# backprojection
backprojected = radon.backward(filtered_sinogram)

# apply image_model to backprojected images
y = image_model(backprojected)

# backward works as usual
loss = criterion(y, x)
loss.backward()

def print_tensor(tensor:torch.Tensor, tensor_name):
    print(f'{tensor_name}: ', tuple(tensor.shape), tensor.min(), tensor.max())

print_tensor(x, 'X')
print_tensor(sinogram, 'Sino')
print_tensor(filtered_sinogram, 'Filtered')
print_tensor(backprojected, 'Backproj')
print_tensor(y, 'Y')
