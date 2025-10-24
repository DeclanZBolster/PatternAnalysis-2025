"""
“train.py" containing the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training
"""

# import torch

# from modules import *
# from dataset import *

# device = torch.device("cuda" if torch.cuda.is_available else "cpu")

# ## Used for processing the images.
# def showImage(img):
#     import matplotlib.pyplot as plt
#     import matplotlib.image as mpimg

#     # Display the image
#     plt.imshow(img)
#     plt.title('My Image') # Optional: Add a title to the image
#     plt.axis('off') # Optional: Turn off axis labels and ticks
#     plt.show()

# ## Hyper-parameters
# batchSize = 1
# epoch = 1

# # ## Training
# # trainDataLoader = Load.trainDataLoader(batchSize)
# # class training():
    
# #     def loadModel(modelPath):
# #         model = VQVAE()


import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 
from dataset import Load 
from modules import Encoder, Decoder, VectorQuantiser, VQVAE

from skimage.metrics import structural_similarity as ssim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batchSize = 8
numEpochs = 5-4
learningRate = 1e-3
inChannels = 64
numEmbeddings = 512
embeddingDim = 64
outChannels = 3
targetShape = (256, 128)

trainBestSSIM = 0;
valBestSSIM = 0; ## Might do this as a list of the best ones?


validationLoss = 0
validation_SSIM = 0

## For saving best model
bestModelSavedDir = "C:/Users/decla/Desktop/2025/Semester2/COMP3710/A3_savedModels"
bestValLoss = float('inf')



## Loading the data
train_loader = Load.loadTrainData(batchSize, target_shape=targetShape)
validate_loader = Load.loadValidateData(batchSize, target_shape=targetShape)

## Initiliasing the model
encoder = Encoder(inChannels).to(device)
decoder = Decoder(in_channel=inChannels*8, out_channel=outChannels).to(device)
vqLayer = VectorQuantiser(num_embeddings=numEmbeddings, embedding_dim=embeddingDim).to(device)
model = VQVAE(encoder, decoder, vqLayer).to(device)

## The optimiser and loss
optimiser = optim.Adam(model.parameters(), lr=learningRate)
criterion = nn.MSELoss()

## The training loop

print("Undergoing Testing...")

for epoch in range(numEpochs):
    model.train()
    totalLoss = 0
    totalReconLoss = 0
    totalVqLoss = 0

    loop = tqdm(train_loader, leave=False)
    for batch in loop:
    # for i in range(epoch):
        # torch.Size([5, 1, 256, 128])
        # batch = torch.randn(5,1,256,128)
        batch = batch.to(device)

        ## look at reference 9 training code when writing this section
        optimiser.zero_grad()
        reconTrain, vq_loss, _ = model(batch)
        recon_loss = criterion(reconTrain, batch)
        loss = recon_loss + vq_loss
        
        loss.backward()
        optimiser.step()


        ## Need to calculate SSIM here with this.
        ssimScore_train = ssim(batch, reconTrain, channel_axis=1)
        print(f"epoch: {epoch} Vector quantised loss {vq_loss}, reconstruction loss: {recon_loss} ssim score: {ssimScore_train}")
    

        ## How do I store the weights here?
        ## Is it fine to find the best one,
        ## or should I use a subset, if so
        if (ssimScore_train > trainBestSSIM) :
            trainBestSSIM_train = ssimScore_train

        ## This is when validation occurs.
        if (epoch % 20 == 0):
            print("validation instance: " + (epoch / 20))
            model.eval()
            validationLoss = 0
            validation_SSIM = 0

            with torch.no_grad():
                for batch in validate_loader:
                    batch = batch.to(device)
                    reconValid, _ = model(batch)
                    validation_SSIM = ssim(batch, reconValid)


                    if (validation_SSIM > valBestSSIM) :
                        valBestSSIM = validation_SSIM
                    

        

        totalLoss += loss.item()
        totalReconLoss += recon_loss.item()
        totalVqLoss += vq_loss.item()
        totalSSIMScore += ssimScore_train

    avgTrainLoss = totalLoss / len(train_loader)
    avgReconLoss = avgReconLoss / len(train_loader)
    avgVqLoss = totalVqLoss / len(train_loader)

    averageSSIMScore = totalSSIMScore / len(train_loader)

    print("average trainining loss: " + avgTrainLoss)
    print("average reconstuction loss: " + avgReconLoss)
    print("average quantised vector loss: " + avgVqLoss)

    # print()


# ## Validation 

# print("Undergoing Validation...")

# model.eval()
# valLossTotal = 0

# with torch.nograd():

#     for valBatch in validate_loader:
#         valBatch = valBatch.to(device)



## once model is trained, run it through validation to find the one with the best
## SSIM score.

# .save(path)
# .load


