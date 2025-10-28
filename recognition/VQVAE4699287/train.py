## This is currently the best training file we have


"""
train.py — Train a VQ-VAE on NIfTI slices using modules.py and dataset.py

Each epoch saves reconstructed images for visual tracking.
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


# Helper functions

def batch_ssim(x, y, assume_minmax=False):
    """Compute average SSIM for a batch."""
    x_np = x.permute(0, 2, 3, 1).detach().cpu().numpy()
    y_np = y.permute(0, 2, 3, 1).detach().cpu().numpy()
    scores = []
    for xi, yi in zip(x_np, y_np):
        dr = 1.0 if assume_minmax else float(xi.max() - xi.min() or 1.0)
        scores.append(ssim(xi, yi, channel_axis=-1, data_range=dr))
    return float(np.mean(scores))


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


# Main training loop

def main():
    ## Directories


    TRAIN_DIR = "C:/Users/s4699287/Desktop/A3_LocalData/keras_slices_train"
    VAL_DIR = "C:/Users/s4699287/Desktop/A3_LocalData/keras_slices_validate"
    OUT_DIR = os.path.join(
        "C:/Users/s4699287/Desktop/A3_localData/vqvae_outputs",
        datetime.now().strftime("%Y%m%d_%H%M"),
    )

    
    os.makedirs(OUT_DIR, exist_ok=True)

    ## Hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 3e-4
    IN_CHANNELS = 64
    # DECODE_IN_CHANNELS = 256
    EMBEDDING_DIM = 64
    NUM_EMBEDDINGS = 512
    OUT_CHANNELS = 1
    VAL_EVERY = 1
    ASSUME_MINMAX = True  # True if you use min-max normalisation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ## Datasets and loaders
    train_set = MRI_dataset(path=TRAIN_DIR, earlyStop=False)
    val_set = MRI_dataset(path=VAL_DIR, earlyStop=False)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    print(f"Loaded {len(train_set)} training and {len(val_set)} validation images.")

    ## Model
    encoder = Encoder(IN_CHANNELS, EMBEDDING_DIM).to(device)
    decoder = Decoder(in_channel=EMBEDDING_DIM, out_channel=OUT_CHANNELS).to(device)
    vq_layer = VectorQuantiser(num_embeddings=NUM_EMBEDDINGS, embedding_dim=EMBEDDING_DIM).to(device)
    model = VQVAE(encoder, decoder, vq_layer).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    ## Trackers
    train_losses, val_losses, train_ssims, val_ssims = [], [], [], []
    best_val_ssim = -1.0


    ## Training loop
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss, epoch_ssim = 0.0, 0.0

        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")

        for batch in pbar:
            batch = batch.to(device)

            optimizer.zero_grad()
            recon, vq_loss, _ = model(batch)
            recon_loss = criterion(recon, batch)
            loss = recon_loss + vq_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_ssim_val = batch_ssim(batch, recon, assume_minmax=ASSUME_MINMAX)
            epoch_loss += loss.item()
            epoch_ssim += batch_ssim_val
            pbar.set_postfix(loss=loss.item(), ssim=batch_ssim_val)

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_ssim = epoch_ssim / len(train_loader)
        train_losses.append(avg_train_loss)
        train_ssims.append(avg_train_ssim)
        print(f"[Train] Loss={avg_train_loss:.4f}, SSIM={avg_train_ssim:.4f}")

        ## Validation 
        if epoch % VAL_EVERY == 0:
            model.eval()
            val_loss, val_ssim_total = 0.0, 0.0
            with torch.no_grad():
                for val_batch in tqdm(val_loader, desc="Validating", leave=False):
                    val_batch = val_batch.to(device)
                    val_recon, val_vq_loss, _ = model(val_batch)
                    vloss = criterion(val_recon, val_batch) + val_vq_loss
                    val_loss += vloss.item()
                    val_ssim_total += batch_ssim(val_batch, val_recon, assume_minmax=ASSUME_MINMAX)

                avg_val_loss = val_loss / len(val_loader)
                avg_val_ssim = val_ssim_total / len(val_loader)
                val_losses.append(avg_val_loss)
                val_ssims.append(avg_val_ssim)

                print(f"[Val] Loss={avg_val_loss:.4f}, SSIM={avg_val_ssim:.4f}")

                ## Save reconstructions each epoch
                save_path = os.path.join(OUT_DIR, f"epoch_{epoch:03d}_recons.png")
                save_recon_grid(val_batch, val_recon, save_path)
                print(f"Saved sample reconstructions to {save_path}")

                if avg_val_ssim > best_val_ssim:
                    best_val_ssim = avg_val_ssim
                    torch.save(model.state_dict(), os.path.join(OUT_DIR, "vqvae_best.pth"))

                    #############################################################################
                    ## Added this here for separate saving of the path to use for the predict.py file
                    torch.save(model.state_dict(), "model.path")
                    #############################################################################

                    print(f"New best model saved (SSIM={best_val_ssim:.4f})")

    ## Save training curves
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "vqvae_last.pth"))
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend(); plt.xlabel("Epoch"); plt.title("Loss"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png")); plt.close()

    plt.figure()
    plt.plot(train_ssims, label="Train SSIM")
    plt.plot(val_ssims, label="Val SSIM")
    plt.legend(); plt.xlabel("Epoch"); plt.title("SSIM"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ssim_curve.png")); plt.close()

    print(f"\nTraining complete. Outputs saved in: {OUT_DIR}")


## Run entry point

if __name__ == "__main__":
    main()