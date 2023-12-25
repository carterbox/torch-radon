import os
import numpy as np
import torch
from PIL import Image
from torch_radon import ParallelBeam, FanBeam

def to_255_numpy(x: torch.Tensor):
    xmin, xmax = x.min(), x.max()
    y = (x - xmin) / (xmax - xmin) * 255
    y = np.uint8(y.detach().cpu().numpy())
    return y

device = torch.device('cuda')
img = np.load("phantom.npy")
image_size = img.shape[0]
det_count = int(image_size * 1.5)  # a det_count close to image_size will result in a circle artifact
n_angles = image_size

output_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(output_dir, 'example_output')
os.makedirs(output_dir, exist_ok=True)

angles = np.linspace(0, np.pi, n_angles, endpoint=False)
parallel_radon = ParallelBeam(det_count, angles)
fanbeam_radon = FanBeam(det_count, angles*2)
x = torch.FloatTensor(img).to(device)

with torch.no_grad():
    sinogram_parallel = parallel_radon.forward(x)
    sinogram_filtered_parallel = parallel_radon.filter_sinogram(sinogram_parallel)
    fbp_parallel = parallel_radon.backward(sinogram_filtered_parallel)
    print("Parallel-beam FBP error", torch.norm(x - fbp_parallel).item())

Image.fromarray(to_255_numpy(x)).save(os.path.join(output_dir, 'x.png'))
Image.fromarray(to_255_numpy(sinogram_parallel)).save(os.path.join(output_dir, 'sino_parallel.png'))
Image.fromarray(to_255_numpy(sinogram_filtered_parallel)).save(os.path.join(output_dir, 'sino_filtered_parallel.png'))
Image.fromarray(to_255_numpy(fbp_parallel)).save(os.path.join(output_dir, 'y_parallel.png'))

with torch.no_grad():
    sinogram_fan = fanbeam_radon.forward(x)
    sinogram_filtered_fan = fanbeam_radon.filter_sinogram(sinogram_fan)
    fbp_fan = fanbeam_radon.backward(sinogram_filtered_fan)
    print("Fan-beam FBP error", torch.norm(x - fbp_fan).item())
    
Image.fromarray(to_255_numpy(sinogram_fan)).save(os.path.join(output_dir, 'sino_fan.png'))
Image.fromarray(to_255_numpy(sinogram_filtered_fan)).save(os.path.join(output_dir, 'sino_filtered_fan.png'))
Image.fromarray(to_255_numpy(fbp_fan)).save(os.path.join(output_dir, 'y_fan.png'))
