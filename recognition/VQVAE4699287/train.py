"""
“train.py" containing the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training
"""

import torch

from modules import *
from dataset import *

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

## Used for processing the images.
def showImage(img):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    # Display the image
    plt.imshow(img)
    plt.title('My Image') # Optional: Add a title to the image
    plt.axis('off') # Optional: Turn off axis labels and ticks
    plt.show()

## Hyper-parameters
batchSize = 1
epoch = 1

# ## Training
# trainDataLoader = Load.trainDataLoader(batchSize)
# class training():
    
#     def loadModel(modelPath):
#         model = VQVAE()


