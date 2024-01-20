import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_radon import FanBeam, Volume2D


class TestDataset(Dataset):
    def __init__(self):
        super().__init__()
        img = np.load('phantom.npy')
        
        imgs = [
            img,
            np.fliplr(img).copy(),
            np.rot90(img, k=1).copy(),
            np.rot90(img, k=2).copy(),
            np.rot90(img, k=3).copy(),
        ]
        self.imgs = [_ for _ in imgs] + [(1.0 - _.copy()) for _ in imgs]
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        x = torch.from_numpy(img).unsqueeze(0).float()
        return x


class TestNet(nn.Module):
    def __init__(self, channels=3, img_size=(512, 512)) -> None:
        super().__init__()
        # Network parameters
        self.sino_model = nn.Sequential(
            nn.Conv2d(1, channels, 5, padding=2),
            nn.Conv2d(channels, 1, 5, padding=2)
        )
        self.img_model = nn.Conv2d(2, 1, 3, padding=1).to(device)
        
        # Tomography parameters
        self.det_count = 672
        self.full_angles = np.linspace(0, np.pi * 2, 360, endpoint=False)
        self.volume = Volume2D()
        self.volume.set_size(img_size[0], img_size[1])
        self.radon = FanBeam(self.det_count, angles=self.full_angles, volume=self.volume)
    
    def forward_sino(self, sino):
        print('>>> Forwarding sino net.')
        return self.sino_model(sino)
    
    def forward_img(self, img1, img2):
        print('>>> Forwarding img net.')
        img = torch.cat([img1, img2], dim=1)
        return self.img_model(img)
    
    def projection(self, img, angles=None, filter=True):
        angles = self.full_angles if angles is None else angles

        sino = self.radon.forward(img, angles=angles)
        if filter:
            sino = self.radon.filter_sinogram(sino)
        return sino
    
    def backprojection(self, sino, angles=None):
        angles = self.full_angles if angles is None else angles

        img = self.radon.backward(sino, angles=angles)
        return img
    
    def forward(self, sino, angles=None):
        img = self.backprojection(sino, angles=angles)
        sino_pred = self.forward_sino(sino)
        img_sino = self.backprojection(sino_pred, angles=angles)
        img_pred = self.forward_img(img, img_sino)
        return sino_pred, img_sino, img_pred



if __name__ == '__main__':
    import os
    import torch.optim as optim
    from torch.utils.data import DataLoader
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # set your own GPU id
    device = torch.device('cuda')
    dataset = TestDataset()
    loader = DataLoader(dataset, batch_size=2, num_workers=4, pin_memory=True, shuffle=True)
    img_size = dataset.imgs[0].shape
    net = TestNet(img_size=img_size).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    angles = torch.linspace(0, np.pi * 2, 360, requires_grad=False).float().to(device)
    
    for i, data in enumerate(loader):
        print(f'Batch: {i}')
        with torch.no_grad():
            sino = net.projection(data.to(device).detach(), angles=angles).detach()
            img = net.backprojection(sino, angles=angles).detach()
        
        sino_pred, _, img_pred = net(sino, angles=angles)
        
        loss1 = F.l1_loss(sino, sino_pred)
        loss2 = F.l1_loss(img, img_pred)
        loss = loss1 + loss2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
