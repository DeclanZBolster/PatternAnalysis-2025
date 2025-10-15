import torch.nn as nn


class VQVAE(nn.module):
    def __init__(self, z_dim=50):
        super().__init()
        # Encoder
        self.enc_conv = nn.Sequential(

            ## necessary image convolution
            nn.Conv2d(1, 32, 4, 2, 1), ## 64->32
            nn.BatchNorm2d(32),
            nn.ReLU(),

            ## Dropout to prevent overfitting
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, 4, 2, 1), ## 32->16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            

            nn.Conv2d(64, 128, 4, 2, 1), ## 16->6
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 256, 4, 2, 1), ## 8->4
            nn.ReLU(),

            ## need to check if this is necessary between
            ## image generation
            nn.Flatten(),
        )

        ## this is the intermittent point where the 
        ## codebook is implemented