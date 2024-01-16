import numpy as np
import torch
from torch_radon import FanBeam, Volume2D
from PIL import Image


def to_255(x, xmin=None, xmax=None):
    xmin = x.min() if xmin is None else xmin
    xmax = x.max() if xmax is None else xmax
    return (x - xmin) / (xmax - xmin) * 255


if __name__ == '__main__':
    img = Image.open('/home/clma/projects/mar/files/flow.png').resize((256, 256))
    img = np.array(img)[:,:,0]
    x = torch.from_numpy(img).float()
    x = x / 255
    angles = torch.tensor(np.linspace(0, 2*np.pi, 360, False)).float()
    volume = Volume2D()
    volume.set_size(x.shape[0], x.shape[1])
    radon = FanBeam(672, angles, volume=volume)
    proj = radon.forward(x.cuda(), angles.cuda())
    rec = radon.backward(proj, angles.cuda())
    print(x.min(), x.max(), x.shape)
    print(proj.min(), proj.max(), proj.shape)
    print(rec.min(), rec.max(), rec.shape)
    
    Image.fromarray(np.uint8(to_255(proj.cpu().numpy()))).save('examples/sino_fan_torchradon2.png')
    Image.fromarray(np.uint8(to_255(rec.cpu().numpy()))).save('examples/recon_fan_torchradon2.png')

