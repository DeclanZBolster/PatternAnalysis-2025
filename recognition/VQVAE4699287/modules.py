import torch.nn as nn


## Might be easier to have them as classes
class Encoder(nn.module):
    def __init__(self, z_dim=50):
        super().__init()
        self.z_dim = z_dim



class Decoder(nn.module):
    def __init__(self, z_dim=50):
        super().__init()
        self.z_dim = z_dim



class VQVAE(nn.module):
    def __init__(self, z_dim=50):
        super().__init()
        self.z_dim = z_dim
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

        # Decoder
        self.fc = nn.Linear(z_dim, 256*4*4)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), #4->8
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1), #8->16
            nn.BatchNorm2d(128), 
            nn.ReLU(), 

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),

            nn.Sigmoid()

        )


