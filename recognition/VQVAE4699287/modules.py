import torch.nn as nn



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init()
    



    def forward()

## Might be easier to have them as classes
class Encoder(nn.module):
    # kernel_size, stride, and padding can just be specified, and doesn't have
    # to be given
    def __init__(self, in_channel, out_channel):
        super().__init()

        conv1 = nn.Conv2d(1, in_channel, kernel_size=4,
                          stride=2, padding=1)
        batchNorm1 = nn.BatchNorm2d(in_channel)

        ## Need to consider how to include residual block in this properly
        resBlock1 = ResidualBlock(in_channel)

        conv2 = nn.Conv2d(in_channel, in_channel*2, kernel_size=4,
                          stride=2, padding=1)
        in_channel *= 2;
        batchNorm2 = nn.BatchNorm2d(in_channel)

        conv3 = nn.Conv2d(in_channel, in_channel*2, kernel_size=4,
                          stride=2, padding=1)
        in_channel *= 2;
        batchNorm3 = nn.BatchNormwd(in_channel)

        conv4 = nn.Conv2d(in_channel, in_channel*2, kernel_size=4,
                          stride=2, padding=1)
        
        self.relu == nn.ReLU()
        

    def forward(self, x):

        x = conv1(x)
        x = 

class Decoder(nn.module):
    def __init__(self, in_channel, kernel_size, stride,
                 padding, z_dim):
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
            ## Will need to implement this
            # ResidualBlock(32),

            ## Dropout to prevent overfitting
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, 4, 2, 1), ## 32->16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ## Will need to implement this
            # ResidualBlock(64),
            nn.Dropout2d(0.2),
            

            nn.Conv2d(64, 128, 4, 2, 1), ## 16->6
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            ## Will need to implement this
            # ResidualBlock(128),
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 256, 4, 2, 1), ## 8->4
            nn.ReLU(),

            ## need to check if this is necessary between
            ## image generation
            nn.Flatten(),
        )

        ## this is the intermittent point where the 
        ## codebook is implemented

        # CodeBook

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


