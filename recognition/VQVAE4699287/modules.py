
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time


class VQVAE(nn.module):
    def __init__(self, z_dim=50):
        super().__init()
        # Encoder
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), ## 64->32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32),
            nn.Conv2d(32, 64, 4, 2, 1) ## 32->16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 128, 4, 2, 1), ## 
        )