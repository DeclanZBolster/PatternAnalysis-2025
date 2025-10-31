"""
VQ-VAE structure, containing its sub-components.
    - ResidualBlock
    - Encoder
    - Decoder
    - VectorQuantiser
    - VQVAE

Author: Declan Bolster
"""


import torch.nn as nn
import torch
import torch.nn.functional as F

"""
See README for more thorough ResidualBlock structure explanation.
Provides additional CNN depth and allows elements of input to directly reach
the CNN block output - reducing minnor feature loss
"""
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        ## Used reference 1 to write
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.batchNorm1 = nn.InstanceNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.batchNorm2 = nn.InstanceNorm2d(channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batchNorm1(x) ## Data normalisation
        x = self.relu1(x) ## Introduces non-linearity to prevent learning halt
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x += residual ## Keeping input in part of output to prevent minor features being removed
        x = self.relu2(x)
        return x
    

"""
See README for more thorough Encoder structure explanation.
Downsamples image for pixel space to continuous representation
"""
class Encoder(nn.Module):
    def __init__(self, in_channel, embeddingDim):
        super().__init__()

        # Downsample 1
        self.conv1 = nn.Conv2d(1, in_channel, kernel_size=4, stride=2, padding=1)
        self.batchNorm1 = nn.InstanceNorm2d(in_channel) ## Data normalisation
        self.relu1 = nn.ReLU() ## Introduces non-linearity to prevent learning halt
        self.resBlock1 = ResidualBlock(in_channel) ## Additional CCN depth

        # Downsample 2
        self.conv2 = nn.Conv2d(in_channel, in_channel*2, kernel_size=4, stride=2, padding=1)
        in_channel *= 2
        self.batchNorm2 = nn.InstanceNorm2d(in_channel)
        self.relu2 = nn.ReLU()
        self.resBlock2 = ResidualBlock(in_channel)

        # Downsample 3
        self.conv3 = nn.Conv2d(in_channel, in_channel*2, kernel_size=4, stride=2, padding=1)
        in_channel *= 2
        self.batchNorm3 = nn.InstanceNorm2d(in_channel)
        self.relu3 = nn.ReLU()
        self.resBlock3 = ResidualBlock(in_channel)

        # Final projection (no downsampling)
        self.finalConv = nn.Conv2d(in_channel, embeddingDim, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu1(x)
        x = self.resBlock1(x)

        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.relu2(x)
        x = self.resBlock2(x)

        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = self.relu3(x)
        x = self.resBlock3(x)

        x = self.finalConv(x)
        return x

"""
See README for more thorough Decoder structure explanation.
Upsamples codeBook vector most resembling the Encoder to pixel space (image)
"""
class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel=1):
        super().__init__()

        # Upsample 1
        self.convTran1 = nn.ConvTranspose2d(in_channel, in_channel//2,
                                            kernel_size=4, stride=2, padding=1)
        in_channel //= 2
        self.batchNorm1 = nn.InstanceNorm2d(in_channel)
        self.relu1 = nn.ReLU()
        self.resBlock1 = ResidualBlock(in_channel)

        # Upsample 2
        self.convTran2 = nn.ConvTranspose2d(in_channel, in_channel//2,
                                            kernel_size=4, stride=2, padding=1)
        in_channel //= 2
        self.batchNorm2 = nn.InstanceNorm2d(in_channel)
        self.relu2 = nn.ReLU()
        self.resBlock2 = ResidualBlock(in_channel)

        # Upsample 3
        self.convTran3 = nn.ConvTranspose2d(in_channel, out_channel,
                                            kernel_size=4, stride=2, padding=1)
        
        ## Normalising output to [0,1]
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
        x = self.sig(x)
        return x


"""
See README for more thorough VectorQuantiser and CodeBook structure explanation.
Quantises the latent space via the vector codeBook, of which is trained to possess
a dictionary with vectors most similar to the output of the Encoder
"""
# Reference 3 and 4
## takes output from the encoder 
class VectorQuantiser(nn.Module):
    ## beta = 0.25 was used in the paper from reference 3,
    ## paper said [0.1, 1] all worked
    ## Typer signature below from reference 5.
    def __init__(self, num_embeddings, embedding_dim, beta=0.4):
        super().__init__()
        self.num_embeddings = num_embeddings ## Is the number of vectors present in the codeBook
        self.embedding_dim = embedding_dim ## Dimensionality of each vector in the codeBook
        self.beta = beta
        ## Reference 6 used here
        self.codeBook = nn.Embedding(num_embeddings, embedding_dim)
        ## Creating a uniform, random distribution of the weights
        self.codeBook.weight.data.uniform_(-1/embedding_dim,
                                           1/embedding_dim)

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

        ## Straight-through resonator (see README for explanaiton)
        ## From reference 6, and mentioned in video of reference 2
        # quantisedVector = x + (x - quantisedVector).detach()
        quantisedVector = x + (quantisedVector - x).detach()


        ## Need to account for losses
        ## Using expanded equation 3 from refernce 3 and inspired by implementation
        ## from reference 7 
        latentCommitLoss = self.beta * F.mse_loss(x.detach(), quantisedVector)
        codeBookLoss = F.mse_loss(x, quantisedVector.detach())
        loss = latentCommitLoss + codeBookLoss

        ## Using reference 5 for expected return of the vectorQuantiser
        # return quantisedVector.permute(0, 2, 3, 1).contiguous(), loss, codeBookIndex
        return quantisedVector.permute(0, 3, 1, 2).contiguous(), loss, codeBookIndex


"""
The cumulative VQ-VAE structure.
See README for more thorough structure explanation.
"""
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
        ## The quantiser is return the vector most similar to the encoder output, the loss while this is achieved, and 
        # the codeBook indices that corresponds to it, respectively.
        quantisedOutput, quantiseLoss, codeBookIndices = self.quantiser(encoderOutput)
        ## The decoder recieves this vector most similar to the Encoder output
        decoderOutput = self.decoder(quantisedOutput)
        ## VQ-VAE then returns the decoder output, the loss of the codeBook, and the codeBook indices
        return decoderOutput, quantiseLoss, codeBookIndices
