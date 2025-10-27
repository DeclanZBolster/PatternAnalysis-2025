"""
“predict.py" showing example usage of your trained model. Print out any results and / or provide visu-
alisations where applicable
"""
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
numEpochs = 20 - 19
learningRate = 1e-3
inChannels = 64
numEmbeddings = 512
embeddingDim = 64
outChannels = 3 ## Cause colour lol
targetShape = (256, 128)

## Only loading the weights from that path (the only thing that is stored)
savedModel = torch.load("model.path", weights_only=True)
## Need to recreate the VQVAE with the same parameters as the one you have saved.

predict_loader = Load.loadTestData(batchSize, target_shape=targetShape)

encoder = Encoder(inChannels, embeddingDim).to(device)
decoder = Decoder(in_channel=embeddingDim, out_channel=outChannels).to(device)
vqLayer = VectorQuantiser(num_embeddings=numEmbeddings, embedding_dim=embeddingDim).to(device)
model = VQVAE(encoder, decoder, vqLayer).to(device)
## Load in the model with everything that has been trained on the in the train.py
model.load_state_dict(savedModel)



print("Undergoing testing...")


# for epoch in range(epoch):