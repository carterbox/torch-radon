import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_radon import FanBeam, Volume2D
from torch.utils.data import DataLoader, Dataset


class TestDataset(Dataset):
    """A dataset of 2D phantoms. Shape is (N, W, H)"""

    def __init__(self):
        super().__init__()
        img = np.load("phantom.npy")

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
    def __init__(
        self,
        channels: int,
        width: int,
        angles: torch.Tensor,
        det_count: int,
    ):
        super().__init__()
        self.sino_model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=channels,
                kernel_size=5,
                padding=2,
            ),
            nn.Conv2d(
                in_channels=channels,
                out_channels=1,
                kernel_size=5,
                padding=2,
            ),
        )
        self.img_model = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )
        volume = Volume2D()
        volume.set_size(width, width)
        self.radon = FanBeam(
            det_count,
            angles,
            volume=volume,
        )

    def forward(self, sino):
        img = self.radon.backward(
            sino,
        )
        print(">>> Forwarding sino net.")
        sino_pred = self.sino_model(sino)
        img_sino = self.radon.backward(
            sino_pred,
        )
        print(">>> Forwarding img net.")
        img_pred = self.img_model(torch.cat([img, img_sino], dim=1))
        return sino_pred, img_sino, img_pred


if __name__ == "__main__":
    device = torch.device("cuda")
    dataset = TestDataset()
    example_data = next(iter(dataset))
    assert example_data.shape == (1, 512, 512), example_data.shape
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )
    example_batch = next(iter(loader))
    assert example_batch.shape == (2, 1, 512, 512), example_batch.shape
    angles = torch.linspace(0, np.pi * 2, 360, requires_grad=False).float().to(device)
    net = TestNet(
        channels=example_batch.shape[-3],
        width=example_batch.shape[-1],
        angles=angles,
        det_count=672,
    ).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    for i, phantom in enumerate(loader):
        print(f"Batch: {i}")

        phantom = phantom.to(device)

        with torch.no_grad():
            sino = net.radon.forward(phantom)
            bp = net.radon.backward(sino)

        sino_pred, _, img_pred = net(sino)

        loss1 = F.l1_loss(sino, sino_pred)
        loss2 = F.l1_loss(bp, img_pred)
        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
