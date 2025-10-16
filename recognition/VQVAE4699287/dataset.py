"""
“dataset.py" containing the data loader for loading and preprocessing your data
"""

import torch
import glob

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

def loadTrainData(batchSize):
    self.batchSize = batchSize
    slices_trainPath = "C:\Users\decla\Desktop\2025\Semester2\COMP3710\A3_VQVAE\keras_slices_data\keras_slices_train"
    ## sorting implementation used from reference 9
    slicesTrainList = sorted(glob.glob(f"{train_folder_path}/**.nii.gz", recursive=True))

def loadValidateData(batchSize):
    self.batchSize = batchSize
    slices_validatePath = "C:\Users\decla\Desktop\2025\Semester2\COMP3710\A3_VQVAE\keras_slices_data\keras_slices_validate"
    slicesValidateList = sorted(glob.glob(f"{train_folder_path}/**.nii.gz", recursive=True))

def loadTestData(batchSize):
    self.batchSize = batchSize
    slices_testPath = "C:\Users\decla\Desktop\2025\Semester2\COMP3710\A3_VQVAE\keras_slices_data\keras_slices_test"
    slicesTestList = sorted(glob.glob(f"{train_folder_path}/**.nii.gz", recursive=True))





# slices_trainPath = "C:\Users\decla\Desktop\2025\Semester2\COMP3710\A3_VQVAE\keras_slices_data\keras_slices_train"
# slices_validatePath = "C:\Users\decla\Desktop\2025\Semester2\COMP3710\A3_VQVAE\keras_slices_data\keras_slices_validate"
# slices_testPath = "C:\Users\decla\Desktop\2025\Semester2\COMP3710\A3_VQVAE\keras_slices_data\keras_slices_test"

## Inspired by reference 9 code in their dataset.py for loading sub-files with the ".nii.gz" file
## suffixes stored in subfolders.

# slicesTrainList = sorted(glob.glob(f"{train_folder_path}/**.nii.gz", recursive=True))
# slicesValidateList = sorted(glob.glob(f"{train_folder_path}/**.nii.gz", recursive=True))
# slicesTestList = sorted(glob.glob(f"{train_folder_path}/**.nii.gz", recursive=True))
