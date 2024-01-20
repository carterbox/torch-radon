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
n_angles = 720
print('Detector count:', det_count)
print('Number of angles:', n_angles)

output_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(output_dir, 'example_output')
os.makedirs(output_dir, exist_ok=True)

angles = np.linspace(0, np.pi, n_angles, endpoint=False)
parallel_radon = ParallelBeam(det_count, angles)
fanbeam_radon = FanBeam(det_count, angles*2)
x = torch.FloatTensor(img).to(device)

with torch.no_grad():
    s_par = parallel_radon.forward(x)
    print("Parallel-beam sinogram info: ", s_par.shape, s_par.min(), s_par.max())
    s_par_filt = parallel_radon.filter_sinogram(s_par)
    x_par = parallel_radon.backward(s_par_filt)
    print("Parallel-beam FBP info: ", x_par.shape, x_par.min(), x_par.max())
    print("Parallel-beam FBP error: ", ((x - x_par)**2).mean())


Image.fromarray(to_255_numpy(x)).save(os.path.join(output_dir, 'orig.png'))
Image.fromarray(to_255_numpy(s_par)).save(os.path.join(output_dir, 'sino_fan_torchradonv2.png'))
Image.fromarray(to_255_numpy(s_par_filt)).save(os.path.join(output_dir, 'sino_fan_filt_torchradonv2.png'))
Image.fromarray(to_255_numpy(x_par)).save(os.path.join(output_dir, 'recon_fan_torchradonv2.png'))
    
with torch.no_grad():
    s_fan = fanbeam_radon.forward(x)
    print("Fan-beam sinogram info: ", s_fan.shape, s_fan.min(), s_fan.max())
    s_fan_filt = fanbeam_radon.filter_sinogram(s_fan)
    x_fan = fanbeam_radon.backward(s_fan_filt)
    print("Fan-beam FBP info: ", x_fan.shape, x_fan.min(), x_fan.max())
    print("Fan-beam FBP error: ", ((x - x_fan)**2).mean())

Image.fromarray(to_255_numpy(s_fan)).save(os.path.join(output_dir, 'sino_fan_torchradonv2.png'))
Image.fromarray(to_255_numpy(s_fan_filt)).save(os.path.join(output_dir, 'sino_fan_filt_torchradonv2.png'))
Image.fromarray(to_255_numpy(x_fan)).save(os.path.join(output_dir, 'recon_fan_torchradonv2.png'))
