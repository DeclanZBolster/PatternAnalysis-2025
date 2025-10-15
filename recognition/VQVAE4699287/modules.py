import torch.nn as nn



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init()
        ## Used reference 1 to write
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x += residual
        x = self.relu2(x)
        return x



## Might be easier to have them as classes
class Encoder(nn.Module):
    # kernel_size, stride, and padding can just be specified, and doesn't have
    # to be given
    def __init__(self, in_channel):
        super().__init__()

        self.conv1 = nn.Conv2d(1, in_channel, kernel_size=4,
                          stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(in_channel)
        self.relu1 = nn.ReLU()
        # self.dropOut1 = nn.Dropout2d(0.2)

        self.resBlock1 = ResidualBlock(in_channel)

        self.conv2 = nn.Conv2d(in_channel, in_channel*2, kernel_size=4,
                          stride=2, padding=1)
        in_channel *= 2
        self.batchNorm2 = nn.BatchNorm2d(in_channel)
        self.relu2 = nn.ReLU()
        # self.dropOut2 = nn.Dropout2d(0.2)

        self.resBlock2 = ResidualBlock(in_channel)

        self.conv3 = nn.Conv2d(in_channel, in_channel*2, kernel_size=4,
                          stride=2, padding=1)
        in_channel *= 2
        self.batchNorm3 = nn.BatchNorm2d(in_channel)
        self.relu3 = nn.ReLU()
        # self.dropOut3 = nn.Dropout2d(0.2)

        self.resBlock3 = ResidualBlock(in_channel)

        self.conv4 = nn.Conv2d(in_channel, in_channel*2, kernel_size=4,
                          stride=2, padding=1)
        
        ## Considering adding the 
        
        self.relu4 = nn.ReLU()
        


    def forward(self, x):

        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu1(x)
        # x = self.dropOut1(x)
        ## Adding ResidualBlock here
        x = self.resBlock1(x)

        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.relu2(x)
        # x = self.dropOut2(x)
        ## Adding ResidualBlock here
        x = self.resBlock2(x)

        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = self.relu3(x)
        # x = self.dropOut3(x)
        x = self.resBlock3(x)

        x = self.conv4(x)
        x = self.relu4(x)

class Decoder(nn.module):
    def __init__(self, in_channel):
        super().__init__()
        
        self.convTran1 = nn.ConvTranspose2d(in_channel, in_channel//2, 
                                            kernel_size=4, stride=2, padding=1) ##4->8
        in_channel //= 2
        self.batchNorm1 = nn.BatchNorm2d(in_channel)
        self.relu1 = nn.ReLU()

        self.convTran2 = nn.ConvTranspose2d(in_channel, in_channel//2,
                                            kernel_size=4, stride=2, padding=1) ##8->16
        in_channel //= 2
        self.batchNorm2 = nn.BatchNorm2d(in_channel)
        self.relu2 = nn.ReLU()
        

    def forward(self, x):

        


# class Decoder(nn.module):
#     def __init__(self, in_channel, kernel_size, stride,
#                  padding, z_dim):
#         super().__init()
#         self.z_dim = z_dim



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


