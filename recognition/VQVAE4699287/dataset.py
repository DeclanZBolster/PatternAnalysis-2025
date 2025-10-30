"""
This file concerns parsing the nifti files for the project, so that they can be individually accessed and uniformly
resized.
"""


import torch
import glob 

import numpy as np
import nibabel as nib
from tqdm import tqdm 

from skimage.transform import resize 

from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

"""
A trimmed version of a method originally provided by Shakes that 
processes nifiti file images.
"""
# Load medical image functions
def load_data_2D(imageNames, normImage=False, dtype=np.float32,
                 early_stop=False):
    """
    Load medical image data from provided file names into a list or 4D array suitable for conv2D.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    Parameters:
        normImage (bool): Normalize the image (0.0 - 1.0).
        categorical (bool): Convert to channel-wise format if True.
        dtype (type): Data type for arrays (default: np.float32).
        getAffines (bool): If True, also return affine matrices.
        early_stop (bool): Stop loading prematurely for quick testing.
    """
    images = []
    # Get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims, remove

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only

        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # sometimes extra dims in HipMRI_study data

        ## Resizing done before normalisation
        inImage = resize(inImage, (256, 128), mode='reflect',
                         preserve_range=True, anti_aliasing=True)
        
        ## Normalisation
        inImage = inImage.astype(dtype)

        if normImage:
            # Normalising the image
            min = inImage.min()
            max = inImage.max()

            inImage = (inImage - min) / (max - min + 1e-8)
          

        if i > 20 and early_stop:
            break

        images.append(inImage)

    
    return images


class MRI_dataset(Dataset):
    ## List of images is path
    def __init__(self, path, earlyStop):
        files = sorted(glob.glob(f"{path}/**.nii.gz", recursive=True))
        self.images = load_data_2D(files, early_stop=earlyStop, normImage=True)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):

        img = self.images[index] ## Grabbing individual image
        img = torch.from_numpy(img).float() ## Creating pytorch tensor representative of image
        img = img.unsqueeze(0) ## Adding additional (1) dimension in order to be greyscale
        return img

    
