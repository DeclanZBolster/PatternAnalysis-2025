"""
“dataset.py" containing the data loader for loading and preprocessing your data
"""

import torch
import glob

import numpy as np
import nibabel as nib
from tqdm import tqdm


from skimage.transform import resize


# ## This method is entirely Shakes', and was provided in the assignment spec
# def to_channels ( arr : np . ndarray , dtype = np . uint8 ) -> np . ndarray :
#     channels = np . unique ( arr )
#     res = np . zeros ( arr . shape + ( len ( channels ) ,) , dtype = dtype )
#     for c in channels :
#         c = int ( c )
#         res [... , c : c +1][ arr == c ] = 1

#     return res
    
# ## This method is entirely Shakes', and was provided in the assignment spec
# # load medical image functions
# def load_data_2D (imageNames , normImage = False , categorical = False , dtype = np . float32 ,
# getAffines = False , early_stop = False ) :
    
#     '''
#     Load medical image data from names , cases list provided into a list for each .
#     This function pre - allocates 4 D arrays for conv2d to avoid excessive memory
#     usage .
#     normImage : bool ( normalise the image 0.0 -1.0)
#     early_stop : Stop loading pre - maturely , leaves arrays mostly empty , for quick 
#     loading and testing scripts .
#     '''

#     affines = []
#     # get fixed size
#     num = len ( imageNames )
#     first_case = nib . load ( imageNames [0]) . get_fdata ( caching = 'unchanged')
#     if len ( first_case . shape ) == 3:
#         first_case = first_case [: ,: ,0] # sometimes extra dims , remove
#     if categorical :
#         first_case = to_channels ( first_case , dtype = dtype )
#         rows , cols , channels = first_case . shape
#         images = np . zeros (( num , rows , cols , channels ) , dtype = dtype )
#     else :
#         rows , cols = first_case . shape
#         images = np . zeros (( num , rows , cols ) , dtype = dtype )
#     for i , inName in enumerate ( tqdm ( imageNames ) ) :
#         niftiImage = nib . load ( inName )
#         inImage = niftiImage . get_fdata ( caching = 'unchanged') # read disk only
#         affine = niftiImage . affine
#         if len ( inImage . shape ) == 3:
#             inImage = inImage [: ,: ,0] # sometimes extra dims in HipMRI_study data

#         inImage = inImage . astype ( dtype )
#         if normImage :
#             # ~ inImage = inImage / np . linalg . norm ( inImage )
#             # ~ inImage = 255. * inImage / inImage . max ()
#             inImage = ( inImage - inImage . mean () ) / inImage . std ()
#         if categorical :
#             inImage = to_channels ( inImage , dtype = dtype )
#             images [i ,: ,: ,:] = inImage
#         else :
#             images [i ,: ,:] = inImage
#         affines . append ( affine )
#         if i > 20 and early_stop :
#             break
#     if getAffines :
#         return images , affines
#     else :
#         return images

## If you ever need categorical one-hot channels
def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

## Load 2D images from NIfTI files with resizing
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, early_stop=False, target_shape=(256,128)):

    affines = []
    num = len(imageNames)
    
    # Pre-allocate array based on first image
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]
    
    # Apply categorical channels if needed
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        images = np.zeros((num, *target_shape, first_case.shape[2]), dtype=dtype)
    else:
        images = np.zeros((num, *target_shape), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        affine = niftiImage.affine

        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]

        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()

        # --- RESIZE ---
        if inImage.shape != target_shape:
            inImage = resize(inImage, target_shape, anti_aliasing=True, preserve_range=True).astype(dtype)

        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)

        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

## Just used for testing.
def showImage(img):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    # Display the image
    plt.imshow(img)
    plt.title('My Image') # Optional: Add a title to the image
    plt.axis('off') # Optional: Turn off axis labels and ticks
    plt.show()
class Load():
    # def _prepare_tensor(images):
    #     """
    #     Convert images to torch tensor and change to channel-first (N, C, H, W)
    #     """
    #     # Ensure images are numpy array
    #     images = np.array(images)
    #     # Transpose from (N, H, W, C) -> (N, C, H, W)
    #     images = np.transpose(images, (0, 3, 1, 2))
    #     return torch.from_numpy(images).float()


    # def loadTrainData(batchSize):
    #     # self.batchSize = batchSize
    #     slices_trainPath = "C:\\Users\\decla\\Desktop\\2025\\Semester2\\COMP3710\\A3_VQVAE\\keras_slices_data\\keras_slices_train"
    #     ## sorting implementation used from reference 9
    #     slicesTrainList = sorted(glob.glob(f"{slices_trainPath}/**.nii.gz", recursive=True))
    #     # print("here")
    #     # print(slicesTrainList[0])
    #     correctedTrainList = load_data_2D(slicesTrainList)
    #     # print("now here")
    #     # print(correctedTrainList[0])
    #     # showImage(correctedTrainList[0])
    #     trainDataLoader = torch.utils.data.DataLoader(correctedTrainList, batch_size=batchSize, shuffle=True)
    #     return trainDataLoader


    @staticmethod
    def _prepare_tensor(images):
        """
        Convert images to torch tensor and change to channel-first (N, C, H, W)
        """
        images = np.array(images)
        if images.ndim == 3:  # grayscale
            images = images[:, np.newaxis, :, :]  # (N, 1, H, W)
        elif images.ndim == 4:  # colored
            images = np.transpose(images, (0, 3, 1, 2))  # (N, C, H, W)
        return torch.from_numpy(images).float()

    @staticmethod
    def loadTrainData(batchSize, normImage=True, early_stop=False, target_shape=(256,128)):
        path = "C:\\Users\\decla\\Desktop\\2025\\Semester2\\COMP3710\\A3_VQVAE\\keras_slices_data\\keras_slices_train"
        files = sorted(glob.glob(f"{path}/**.nii.gz", recursive=True))
        images = load_data_2D(files, normImage=normImage, early_stop=early_stop, target_shape=target_shape)
        tensor_data = Load._prepare_tensor(images)
        return torch.utils.data.DataLoader(tensor_data, batch_size=batchSize, shuffle=True)
    
    ####----
    @staticmethod
    def loadValidateData(batchSize, normImage=True, early_stop=False, target_shape=(256,128)):
        # self.batchSize = batchSize
        path = "C:\\Users\\decla\\Desktop\\2025\\Semester2\\COMP3710\\A3_VQVAE\\keras_slices_data\\keras_slices_validate"
        files = sorted(glob.glob(f"{path}/**.nii.gz", recursive=True))
        images = load_data_2D(files, normImage=normImage, target_shape=target_shape)
        tensor_data = Load._prepare_tensor(images)
        return torch.utils.data.DataLoader(tensor_data, batch_size=batchSize, shuffle=False)

    @staticmethod
    def loadTestData(batchSize, normImage=True, early_stop=False, target_shape=(256,128)):
        # self.batchSize = batchSize
        path = "C:\\Users\\decla\\Desktop\\2025\\Semester2\\COMP3710\\A3_VQVAE\\keras_slices_data\\keras_slices_test"
        files = sorted(glob.glob(f"{path}/**.nii.gz", recursive=True))
        images = load_data_2D(files, normImage=normImage, target_shape=target_shape)
        tensor_data = Load._prepare_tensor(images)
        return torch.utils.data.DataLoader(tensor_data, batch_size=batchSize, shuffle=False)

train_loader = Load.loadTrainData(batchSize=8, target_shape=(256,128))

# for batch in train_loader:
#     print(batch.shape)  # torch.Size([8, 3, 256, 128]) for colored images
#     showImage(batch[0].permute(1,2,0).numpy())  # Correct for colored images
#     break

# train_loader = Load.loadTrainData(batchSize=8, target_shape=(256,128))

# Iterate over batches
for batch in train_loader:
    print(batch.shape)  # e.g., torch.Size([8, 3, 256, 128])
    
    # Show each image manually; next image appears after closing the previous
    for i in range(batch.shape[0]):
        img = batch[i]
        import matplotlib.pyplot as plt
        plt.figure()
        # Convert to HWC for matplotlib
        
        img_np = img.permute(1, 2, 0).numpy()
        plt.imshow(img_np)
        plt.axis('off')
        plt.title(f"Image {i+1}")
        plt.show()  # blocks until you close the window

    break  # remove this if you want to go through all batches

# x = Load.loadTrainData(1)
# showImage(x)
# # for batch in x:
# #     showImage(batch[0])
# print(type(x))





# train_loader = Load.loadTrainData(batchSize=8)
# for batch in train_loader:
#     print(batch.shape)  # e.g., (8, 1, 256, 128)
#     showImage(batch[0].squeeze(0).numpy())
#     break

# # Suppose you already have your train loader
# train_loader = Load.loadTrainData(batchSize=1)  # batch_size=1 for testing

# # Take one batch
# for batch in train_loader:
#     # batch shape: (B, H, W)
#     img = batch[0] if isinstance(batch, list) or isinstance(batch, tuple) else batch[0]
#     img = img.numpy()  # convert to numpy

#     # Normalize to [0,1] for display
#     img = (img - img.min()) / (img.max() - img.min())

#     plt.imshow(img, cmap='gray')
#     plt.axis('off')
#     plt.show()
#     break



# slices_trainPath = "C:\Users\decla\Desktop\2025\Semester2\COMP3710\A3_VQVAE\keras_slices_data\keras_slices_train"
# slices_validatePath = "C:\Users\decla\Desktop\2025\Semester2\COMP3710\A3_VQVAE\keras_slices_data\keras_slices_validate"
# slices_testPath = "C:\Users\decla\Desktop\2025\Semester2\COMP3710\A3_VQVAE\keras_slices_data\keras_slices_test"

## Inspired by reference 9 code in their dataset.py for loading sub-files with the ".nii.gz" file
## suffixes stored in subfolders.

# slicesTrainList = sorted(glob.glob(f"{train_folder_path}/**.nii.gz", recursive=True))
# slicesValidateList = sorted(glob.glob(f"{train_folder_path}/**.nii.gz", recursive=True))
# slicesTestList = sorted(glob.glob(f"{train_folder_path}/**.nii.gz", recursive=True))
