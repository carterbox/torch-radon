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
        x = torch.from_numpy(img).unsqueeze(0).float().contiguous()
        return x


class TestNet(nn.Module):
    def __init__(self, channels=3) -> None:
        super().__init__()
        self.sino_model = nn.Sequential(
            nn.Conv2d(1, channels, 5, padding=2),
            nn.Conv2d(channels, 1, 5, padding=2)
        )
        self.img_model = nn.Conv2d(2, 1, 3, padding=1).to(device)
        self.det_count = 672
    
    def forward_sino(self, sino):
        print('>>> Forwarding sino net.')
        return self.sino_model(sino)
    
    def forward_img(self, img1, img2):
        print('>>> Forwarding img net.')
        img = torch.cat([img1, img2], dim=1)
        return self.img_model(img)
    
    def projection(self, img, angles=None, filter=True):
        if angles is None:
            angles = np.linspace(0, np.pi * 2, 360, endpoint=False)
        
        volume = Volume2D()
        volume.set_size(img.shape[-2], img.shape[-1])  # [B, C, H, W]
        radon = FanBeam(self.det_count, angles, volume=volume)
        sino = radon.forward(img)
        if filter:
            sino = radon.filter_sinogram(sino)
        return sino
    
    def backprojection(self, sino, img_shape, angles=None):
        if angles is None:
            angles = np.linspace(0, np.pi * 2, 360, endpoint=False)

        volume = Volume2D()
        volume.set_size(img_shape[-2], img_shape[-1])  # [H, W]
        radon = FanBeam(self.det_count, angles, volume=volume)
        img = radon.backward(sino)
        return img
    
    def forward(self, sino, img_shape, angles=None):
        img = self.backprojection(sino, img_shape, angles=angles)
        sino_pred = self.forward_sino(sino)
        img_sino = self.backprojection(sino_pred, img_shape, angles=angles)
        img_pred = self.forward_img(img, img_sino)
        return sino_pred, img_sino, img_pred
    
    # ==== The following forward (adding contiguous()) does not work ====
    # def forward(self, sino, img_shape, angles=None):
    #     img = self.backprojection(sino, img_shape, angles=angles)
    #     img = img.contiguous()
    #     sino_pred = self.forward_sino(sino)
    #     img_sino = self.backprojection(sino_pred, img_shape, angles=angles)
    #     img_sino = img_sino.contiguous()
    #     img_pred = self.forward_img(img, img_sino)
    #     return sino_pred, img_sino, img_pred



if __name__ == '__main__':
    import torch.optim as optim
    from torch.utils.data import DataLoader
    
    device = torch.device('cuda')
    dataset = TestDataset()
    loader = DataLoader(dataset, batch_size=2, num_workers=4, pin_memory=True, shuffle=True)
    net = TestNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    angles = torch.linspace(0, np.pi * 2, 360, requires_grad=False).float()
    
    for i, data in enumerate(loader):
        print(f'Batch: {i}')
        img_shape = data.shape[2:]
        with torch.no_grad():
            sino = net.projection(data.to(device).detach(), angles=angles).detach().contiguous()
            img = net.backprojection(sino, img_shape=img_shape, angles=angles).detach().contiguous()
        
        sino_pred, _, img_pred = net(sino, img_shape=img_shape, angles=angles)
        
        loss1 = F.l1_loss(sino, sino_pred)
        loss2 = F.l1_loss(img, img_pred)
        loss = loss1 + loss2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
