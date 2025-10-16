import torch.nn as nn
import torch
import torch.nn.functional as F




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

        return x

class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.convTran1 = nn.ConvTranspose2d(in_channel, in_channel//2, 
                                            kernel_size=4, stride=2, padding=1) ##4->8
        in_channel //= 2
        self.batchNorm1 = nn.BatchNorm2d(in_channel)
        self.relu1 = nn.ReLU()

        self.resBlock1 = ResidualBlock(in_channel)

        self.convTran2 = nn.ConvTranspose2d(in_channel, in_channel//2,
                                            kernel_size=4, stride=2, padding=1) ##8->16
        in_channel //= 2
        self.batchNorm2 = nn.BatchNorm2d(in_channel)
        self.relu2 = nn.ReLU()

        self.resBlock2 = ResidualBlock(in_channel)

        self.convTran3 = nn.ConvTranspose2d(in_channel, in_channel//2,
                                            kernel_size=4, stride=2, padding=1) ## 16->32
        in_channel //= 2
        self.batchNorm3 = nn.BatchNorm2d(in_channel)
        self.relu3 = nn.ReLU()
        
        self.resBlock3 = ResidualBlock(in_channel)

        self.convTran4 = nn.ConvTranspose2d(in_channel, out_channel,
                                            kernel_size=4, stride=2, padding=1) ## 32->64

        #self.relu4 = nn.ReLU()
        self.sig = nn.Sigmoid()


    def forward(self, x):

        x = self.convTran1(x)
        x = self.batchNorm1(x)
        x = self.relu1(x)

        x = self.resBlock1(x)

        x = self.convTran2(x)
        x = self.batchNorm2(x)
        x = self.relu2(x)

        x = self.resBlock2(x)

        x = self.convTran3(x)
        x = self.batchNorm3(x)
        x = self.relu3(x)

        x = self.resBlock3(x)

        x = self.convTran4(x)
        #x = self.relu4(x)
        
        x = self.sig(x)

        return x
    

# Reference 3 and 4
## takes output from the encoder 
class VectorQuantiser(nn.Module):
    ## beta = 0.25 was used in the paper from reference 3
    ## Typer signature below from reference 5.
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        ## Reference 6 used here
        self.codeBook = nn.Embedding(num_embeddings, embedding_dim)
        ## Creating a uniform, random distribution of the weights
        self.codeBook.weight.data.uniform_(-1/num_embeddings,
                                           1/num_embeddings)

    def forward(self, x):

        ## compute which codeBook vector is closest to each
        ## encoder output vector

            ## Comparing each latent vector to all vectors in the codeBook,
            ## so it is flattened into these 4 dimensions
            ## (Reference 4)

            ## Inspired form reference 7 on flattening the incoming
            ## vector to match the expected 4D structure.
        # convert inputs from BCHW -> BHWC, referenced in video from reference 2.
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape

        ## flattening input
        flat_input = x.view(-1, self.embedding_dim)

        ## Using the expanded version of formula 2 from the reference 3 paper
        ## zq(x) = ek, where k = argminj||ze(x) − ej||2, and used direcly in reference 7

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self.codeBook.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.codeBook.weight.t()))
        
        ## Finding the index of the codeBook vector most similar to each vector input
        codeBookIndex = torch.argmin(distances, dim=1)

        ## Grabbing the discrete most-similar vector
        quantisedVector = self.codeBook(codeBookIndex)
        quantisedVector = quantisedVector.view(x_shape)

        ## Straight-through resonator
        ## From reference 6, and mentioned in video of reference 2
        quantisedVector = x + (x - quantisedVector).detach()

        ## Need to account for losses
        ## Using expanded equation 3 from refernce 3 and inspired by implementation
        ## from reference 7
        latentCommitLoss = self.beta * F.mse_loss(x.detach(), quantisedVector)
        codeBookLoss = F.mse_loss(x, quantisedVector.detach())
        loss = latentCommitLoss + codeBookLoss

        ## Using reference 5 for expected return of the vectorQuantiser
        return quantisedVector.permute(0, 2, 3, 1).contiguous(), loss, codeBookIndex


## using reference 8 as inspiration for implementation.
class VQVAE(nn.Module):
    def __init__(self, encoder, decoder,
                 vectorQuantiser):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder 
        self.quantiser = vectorQuantiser

    def forward(self, x):
        
        ## using reference 8 for inspiration on implementation
        encoderOutput = self.encoder(x);
        quantisedOutput, quantiseLoss, codeBookIndices = self.quantiser(encoderOutput)
        decoderOutput = self.decoder(quantisedOutput)

        return decoderOutput, quantiseLoss, codeBookIndices





        ## replace the input with that nearest code vector (discrete vectorisation)

        ## use a straight-through estimator to allow gradients to flow



## Need to determine if this is necessary to include.
class codeBook(nn.Module):
    def __init__self(self):


## Need to implement this. 
class VQVAE(nn.Module):
    def __init__self(self, encoder, decoder, 
                     vectorQuantiser):
        self.encoder = encoder
        self.decoder = decoder 
        self.quantiser = vectorQuantiser

    def forward(self, x):


# class Decoder(nn.module):
#     def __init__(self, in_channel, kernel_size, stride,
#                  padding, z_dim):
#         super().__init()
#         self.z_dim = z_dim



# class VQVAE(nn.module):
#     def __init__(self, z_dim=50):
#         super().__init()
#         self.z_dim = z_dim
#         # Encoder
#         self.enc_conv = nn.Sequential(

#             ## necessary image convolution
#             nn.Conv2d(1, 32, 4, 2, 1), ## 64->32
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             ## Will need to implement this
#             # ResidualBlock(32),

#             ## Dropout to prevent overfitting
#             nn.Dropout2d(0.2),

#             nn.Conv2d(32, 64, 4, 2, 1), ## 32->16
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             ## Will need to implement this
#             # ResidualBlock(64),
#             nn.Dropout2d(0.2),
            

#             nn.Conv2d(64, 128, 4, 2, 1), ## 16->6
#             nn.BatchNorm2d(128),
#             nn.ReLU(), 
#             ## Will need to implement this
#             # ResidualBlock(128),
#             nn.Dropout2d(0.2),

#             nn.Conv2d(128, 256, 4, 2, 1), ## 8->4
#             nn.ReLU(),

#             ## need to check if this is necessary between
#             ## image generation
#             nn.Flatten(),
#         )

#         ## this is the intermittent point where the 
#         ## codebook is implemented

#         # CodeBook

#         # Decoder
#         self.fc = nn.Linear(z_dim, 256*4*4)
#         self.dec_conv = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 4, 2, 1), #4->8
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             nn.ConvTranspose2d(128, 64, 4, 2, 1), #8->16
#             nn.BatchNorm2d(128), 
#             nn.ReLU(), 

#             nn.ConvTranspose2d(64, 32, 4, 2, 1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 1, 4, 2, 1),

#             nn.Sigmoid()

#         )


