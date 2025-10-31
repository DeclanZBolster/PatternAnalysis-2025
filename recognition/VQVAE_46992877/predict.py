"""
This file loads the best model weights from the training and validation
data sets. The test data is then passed to it, and a sample of the
outputs are saved to a local directory.

Includes:
    - batch_ssim
    - save_recon_grid
    - save_batch_images
    - main()

Handles:
    - Generating reconstruction of test set
    - Returning SSIM spread
    - Average loss

Author: Declan Bolster
"""


import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from modules import Encoder, Decoder, VectorQuantiser, VQVAE
from dataset import MRI_dataset

import csv




"""
This method was generated with AI and calculates batch SSIM
"""
def batch_ssim(x, y, assume_minmax=False):
    """Compute average SSIM for a batch."""
    x_np = x.permute(0, 2, 3, 1).detach().cpu().numpy()
    y_np = y.permute(0, 2, 3, 1).detach().cpu().numpy()
    scores = []
    for xi, yi in zip(x_np, y_np):
        dr = 1.0 if assume_minmax else float(xi.max() - xi.min() or 1.0)
        scores.append(ssim(xi, yi, channel_axis=-1, data_range=dr))
    return float(np.mean(scores))

"""
This method was generated with AI and generates comparison image of reconstructed and original images side-by-side.
"""
def save_recon_grid(x, recon, save_path, max_show=8):
    """Save a grid comparing original vs reconstruction."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    b = min(x.size(0), max_show)
    x_np = x[:b].detach().cpu().numpy()
    r_np = recon[:b].detach().cpu().numpy()
    plt.figure(figsize=(6, 2 * b))
    for i in range(b):
        ax = plt.subplot(b, 2, 2 * i + 1)
        ax.imshow(x_np[i, 0], cmap="gray")
        ax.set_title("Original")
        ax.axis("off")
        ax = plt.subplot(b, 2, 2 * i + 2)
        ax.imshow(r_np[i, 0], cmap="gray")
        ax.set_title("Reconstructed")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


"""
This method was generated with AI, optionally saving per-image reconstructions
"""
def save_batch_images(x, recon, base_dir, batch_idx):
    """Optionally save per-image reconstructions for inspection."""
    os.makedirs(base_dir, exist_ok=True)
    b = x.size(0)
    x_np = x.detach().cpu().numpy()
    r_np = recon.detach().cpu().numpy()
    for i in range(b):
        fig = plt.figure(figsize=(4, 2))
        ax = plt.subplot(1, 2, 1)
        ax.imshow(x_np[i, 0], cmap="gray"); ax.set_title("Original"); ax.axis("off")
        ax = plt.subplot(1, 2, 2)
        ax.imshow(r_np[i, 0], cmap="gray"); ax.set_title("Recon"); ax.axis("off")
        out_path = os.path.join(base_dir, f"batch{batch_idx:04d}_img{i:03d}.png")
        plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close(fig)

def main():


    
    ## Local path for test images
    NEW_IMAGES_DIR = "C:/Users/s4699287/Desktop/A3_dataStorage/keras_slices_test"
    
    ## Saving each image to a directory,
    ## as opposed to a subset.
    SAVE_PER_IMAGE = False

    ## The out directory where the reconstruction and data of the model
    ## is saved to.
    ## line generated with AI.
    PREDICT_OUT_DIR = os.path.join(
        "C:/Users/s4699287/Desktop/A3_localData/vqvae_predictionsFromPredict",
        datetime.now().strftime("%Y%m%d_%H%M"),
    )

    ## line generated with AI.
    os.makedirs(PREDICT_OUT_DIR, exist_ok=True)

    ## Hyperparameters
    BATCH_SIZE = 16
    IN_CHANNELS = 64
    EMBEDDING_DIM = 32 
    NUM_EMBEDDINGS = 128 
    OUT_CHANNELS = 1
    ## line of code generated with AI.
    ASSUME_MINMAX = True  # using min-max normalisation
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(IN_CHANNELS, EMBEDDING_DIM).to(device)
    decoder = Decoder(in_channel=EMBEDDING_DIM, out_channel=OUT_CHANNELS).to(device)
    vq_layer = VectorQuantiser(num_embeddings=NUM_EMBEDDINGS, embedding_dim=EMBEDDING_DIM).to(device)
    model = VQVAE(encoder, decoder, vq_layer).to(device)

    
    print(f"Using device: {device}")

    # Test dataset
    predict_set  = MRI_dataset(path=NEW_IMAGES_DIR, earlyStop=False)
    predict_loader = DataLoader(predict_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    print(f"Loaded {len(predict_set)} images to be reconstructed.")
    
    ## Loading the best model
    savedModel = torch.load("model.path", weights_only=True)
    model.load_state_dict(savedModel)


    model.eval()
    criterion = nn.MSELoss(reduction="mean")

    all_loss, all_ssim = 0.0, 0.0
    num_batches = 0
    lowestPredictSSIM = 999
    highestPredictSSIM = 0



    with torch.no_grad():
        pbar = tqdm(predict_loader, desc="Predicting")
        for b_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            recon, _, _ = model(batch)

            loss = criterion(recon, batch).item()
            ssim_score = batch_ssim(batch, recon, assume_minmax=ASSUME_MINMAX)

            if ssim_score < lowestPredictSSIM:
                lowestPredictSSIM = ssim_score

            if ssim_score > highestPredictSSIM:
                highestPredictSSIM = ssim_score

            # logging
            all_loss += loss
            all_ssim += ssim_score
            num_batches += 1
            pbar.set_postfix(mse=loss, ssim=ssim_score)

            # Save one grid from the first batch and then every 10th batch
            ## This code block was generated by the use of AI
            if b_idx % 10 == 0:
                grid_path = os.path.join(PREDICT_OUT_DIR, f"test_batch_{b_idx:04d}_grid.png")
                save_recon_grid(batch, recon, grid_path)
            ## This code block was generayed by the use of AI
            if SAVE_PER_IMAGE:
                per_img_dir = os.path.join(PREDICT_OUT_DIR, "per_image")
                save_batch_images(batch, recon, per_img_dir, b_idx)

    avg_loss = all_loss / max(1, num_batches)
    avg_ssim = all_ssim / max(1, num_batches)

    # Summary
    print(f"Predictions complete. Outputs saved in: {PREDICT_OUT_DIR}")
    print(f"\n[Predict Summary] MSE: {avg_loss:.6f} | SSIM: {avg_ssim:.6f}")
    print(f"\nAvg SSIM: {avg_ssim:.6f}")
    print(f"\nLowest SSIM score: {lowestPredictSSIM}")
    print(f"\nHighest SSIM score: {highestPredictSSIM}")
    print(f"\nAverage loss: {avg_loss}")



if __name__ == "__main__":
    main()