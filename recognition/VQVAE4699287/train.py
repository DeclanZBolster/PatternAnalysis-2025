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

import numpy as np

from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batchSize = 8 - 7
numEpochs = 20
learningRate = 1e-3
inChannels = 64
numEmbeddings = 512
embeddingDim = 64
outChannels = 3 ## Cause colour lol
targetShape = (256, 128)

trainBestSSIM = 0;
valBestSSIM = 0; ## Might do this as a list of the best ones?


validationLoss = 0
validation_SSIM = 0

## For saving best model
bestModelSavedDir = "C:/Users/decla/Desktop/2025/Semester2/COMP3710/A3_savedModels"
import os; os.makedirs(bestModelSavedDir, exist_ok=True)
bestValLoss = float('inf')
bestValSSIM = 0

bestSSIM = 0


## Loading the data
train_loader = Load.loadTrainData(batchSize, target_shape=targetShape)
validate_loader = Load.loadValidateData(batchSize, target_shape=targetShape)

## Initiliasing the model
encoder = Encoder(inChannels, embeddingDim).to(device)
decoder = Decoder(in_channel=embeddingDim, out_channel=outChannels).to(device)
vqLayer = VectorQuantiser(num_embeddings=numEmbeddings, embedding_dim=embeddingDim).to(device)
model = VQVAE(encoder, decoder, vqLayer).to(device)

## The optimiser and loss
optimiser = optim.Adam(model.parameters(), lr=learningRate)
criterion = nn.MSELoss()

## The training loop

trainLosses = []
valLosses = []
ssimScores = []

valSSIM = []

print("Undergoing Testing...")

for epoch in range(numEpochs):
    model.train()
    runningLoss = 0.0
    totalLoss = 0
    totalReconLoss = 0
    totalVQLoss = 0
    totalSSIM = 0

    loop = tqdm(train_loader, leave=False)
    for batch in loop:
    # for i in range(epoch):
        # torch.Size([5, 1, 256, 128])
        # batch = torch.randn(5,1,256,128)
        batch = batch.to(device)

        ## look at reference 9 training code when writing this section
        optimiser.zero_grad()
        reconTrainImages, vq_loss, _ = model(batch)

        recon_loss = criterion(reconTrainImages, batch)
        loss = recon_loss + vq_loss



        ############
        ## This replacing the above two lines because the loss is already calculated 
        ## in the VQVAE vector quantiser
        # vq_loss.backward()
        ############
        loss.backward()
        #########
        ## This above line may then be unnecessary if the vq_loss.backward()
        optimiser.step()

        ########
        totalLoss += loss.item()
        ########
        ## .item() because I have already called .backward()
        # totalLoss += vq_loss.item()

        totalReconLoss += recon_loss.item()
        totalVQLoss += vq_loss.item()


        runningLoss += loss.item() * batch.size(0)

        ## Using the in-built SSIM function, so converting to numpy arrays
        ## This permutation is necssary because the shape of the images
        ## Is currently (batch size, 3, 256, 128)
        ## But the ssim method expects the colour dimension to the be 
        ## at the end of the method.
        ## This is essentially computes the average SSIM over the batch.
        batch_np = batch.permute(0, 2, 3, 1).detach().cpu().numpy()
        recon_np = reconTrainImages.permute(0, 2, 3, 1).detach().cpu().numpy()
        batch_ssims = [ssim(b, r, channel_axis=-1, data_range=1.0) for b, r in zip(batch_np, recon_np)]
        avg_ssim = np.mean(batch_ssims)
        totalSSIM += avg_ssim

        loop.set_postfix(loss=loss.item(), ssim=avg_ssim)

        ## Might do a single batch through the validation here.
        ## Instead of doing validation at the end of every epoch.

        ## Testing for one image here
        # break
    
    avgTrainLoss = totalLoss / len(train_loader)
    avgReconLoss = totalReconLoss / len(train_loader)
    avgVQLoss = totalVQLoss / len(train_loader)
    avgTrainSSIM = totalSSIM / len(train_loader)

    trainLosses.append(avgTrainLoss)
    ssimScores.append(avgTrainSSIM)

    if (avgTrainSSIM > bestSSIM) :
        bestSSIM = avgTrainSSIM
        ## Saving the best model
        torch.save(model.state_dict(), "model.pth")
        # torch.save(model.state_dict(), f"{bestModelSavedDir}/vqvae_best_loss.pt")


    print(f"Epoch: {epoch}")
    print(f"trainLoss: {avgTrainLoss}") 
    print(f"avgReconstructionLoss: {avgReconLoss}") 
    print(f"averageVQLoss: {avgVQLoss}")
    print(f"SSIM: {avgTrainSSIM}") 

    print("Validation...")
    
    if (epoch % 5 == 0):
        model.eval()
        validationLoss = 0
        valLossTotal = 0
        validation_SSIM_Total = 0

        with torch.no_grad():
            for validBatch in validate_loader:
                validBatch = validBatch.to(device)
                reconValidImages, vqValidLoss, _ = model(validBatch)
                valid_recon_loss = criterion(reconValidImages, validBatch)
                validationLoss = valid_recon_loss + vqValidLoss
                valLossTotal += validationLoss.item()

                validBatch_np = validBatch.permute(0, 2, 3, 1).cpu().numpy()
                validRecon_np = reconValidImages.permute(0, 2, 3, 1).cpu().numpy()
                val_ssims = [ssim(b, r, channel_axis=-1, data_range=1.0) for b, r in zip(validBatch_np, validRecon_np)]
                validation_SSIM_Total += np.mean(val_ssims)

        avgValLoss = valLossTotal  / len(validate_loader)
        avgValSSIM = validation_SSIM_Total / len(validate_loader)
        valLosses.append(avgValLoss)
        valSSIM.append(avgValSSIM)

        if (avgValSSIM > bestSSIM):
            bestSSIM = avgValSSIM

            ## Saving the model
            torch.save(model.state_dict(), f"{bestModelSavedDir}/vqvae_best_ssim.pt")

        print(f"ValidationEpoch: {epoch}")
        print(f"ValidationtrainLoss: {avgValLoss}") 
        # print(f"ValidationAvgReconstructionLoss: {avgReconLoss}") 
        # print(f"ValidationAverageVQLoss: {avgVQLoss}")
        print(f"ValidationSSIM: {avgValSSIM}") 

        model.train()
                

# # Original
# ax = plt.subplot(2, n, i + 1)
# plt.imshow(test_dataset[i].squeeze(), cmap="gray")
# plt.axis("off")
# # Reconstruction
# ax = plt.subplot(2, n, i + 1 + n)
# plt.imshow(reconstructions[i].squeeze(), cmap="gray")

plt.figure()
plt.plot(trainLosses, label="Train Loss")
plt.plot(valLosses, label="Val Loss")
plt.legend(); plt.title("Loss"); plt.xlabel("Epoch (val every 5)"); plt.tight_layout(); plt.show()

plt.figure()
plt.plot(ssimScores, label="Train SSIM")
# If you also want val SSIM, track it in a list and plot similarly
plt.legend(); plt.title("SSIM"); plt.xlabel("Epoch"); plt.tight_layout(); plt.show()

plt.figure()
plt.plot(ssimScores, label="Train SSIM")
plt.plot(valSSIM,  label="Val SSIM")
plt.legend(); plt.title("SSIM"); plt.xlabel("Epoch (val every 5)"); plt.tight_layout(); plt.show()
            

        # ## Need to calculate SSIM here with this.
        # ssimScore_train = ssim(batch, reconTrainImages, channel_axis=1)
        # print(f"epoch: {epoch} Vector quantised loss {vq_loss}, reconstruction loss: {recon_loss} ssim score: {ssimScore_train}")
    

        ## How do I store the weights here?
        ## Is it fine to find the best one,
        ## or should I use a subset, if so
        # if (ssimScore_train > trainBestSSIM) :
        #     trainBestSSIM_train = ssimScore_train

        ## This is when validation occurs.
    # if (epoch % 4 == 0):
    #     print("validation instance: " + (epoch / 20))
    #     model.eval()
    #     validationLoss = 0
    #     validation_SSIM = 0

    #     with torch.no_grad():
    #         for batch in validate_loader:
    #             batch = batch.to(device)
    #             reconValidImages, _ = model(batch)
    #             validation_SSIM = ssim(batch, reconValidImages)


    #             if (validation_SSIM > valBestSSIM) :
    #                 valBestSSIM = validation_SSIM
                    

        

    # totalLoss += loss.item()
    # totalReconLoss += recon_loss.item()
    # totalVqLoss += vq_loss.item()
    # totalSSIMScore += ssimScore_train

    # avgTrainLoss = totalLoss / len(train_loader)
    # avgReconLoss = totalReconLoss / len(train_loader)
    # avgVqLoss = totalVqLoss / len(train_loader)

    # averageSSIMScore = totalSSIMScore / len(train_loader)

    # print("average trainining loss: " + avgTrainLoss)
    # print("average reconstuction loss: " + avgReconLoss)
    # print("average quantised vector loss: " + avgVqLoss)

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


