import os
from datetime import datetime
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from modules import Encoder, Decoder, VectorQuantiser, VQVAE
from dataset import MRI_dataset


# Paths & options

TEST_DIR = "C:/Users/s4699287/Desktop/A3_LocalData/keras_slices_test"

OUT_DIR = os.path.join(
    "C:/Users/s4699287/Desktop/A3_localData/vqvae_predictions",
    datetime.now().strftime("%Y%m%d_%H%M"),
)
os.makedirs(OUT_DIR, exist_ok=True)

SAVE_PER_IMAGE = False  # set True to save each recon as its own PNG


# Hyperparams (match training)
BATCH_SIZE = 16
IN_CHANNELS = 64
EMBEDDING_DIM = 64
NUM_EMBEDDINGS = 512
OUT_CHANNELS = 1
ASSUME_MINMAX = True  # True if inputs were min–max in [0,1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------
# Helpers
# -------------------------
def batch_ssim(x, y, assume_minmax=False):
    """Compute average SSIM for a batch (expects [B,1,H,W])."""
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



# Building model & load weights
encoder = Encoder(IN_CHANNELS, EMBEDDING_DIM).to(device)
decoder = Decoder(in_channel=EMBEDDING_DIM, out_channel=OUT_CHANNELS).to(device)
vq_layer = VectorQuantiser(num_embeddings=NUM_EMBEDDINGS, embedding_dim=EMBEDDING_DIM).to(device)
model = VQVAE(encoder, decoder, vq_layer).to(device)

## Saving the model
savedModel = torch.load("model.path", weights_only=True)
model.load_state_dict(savedModel)

model.eval()
criterion = nn.MSELoss(reduction="mean")

## Accessing the test data
test_set = MRI_dataset(path=TEST_DIR, earlyStop=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
print(f"Loaded {len(test_set)} test images.")


## Undergoing prediction based on the test images the model has not seen.
all_loss, all_ssim = 0.0, 0.0
num_batches = 0

csv_path = os.path.join(OUT_DIR, "test_metrics.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow(["batch_idx", "mse_loss", "ssim"])

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Predicting")
        for b_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            recon, _, _ = model(batch)

            loss = criterion(recon, batch).item()
            ssim_score = batch_ssim(batch, recon, assume_minmax=ASSUME_MINMAX)

            # logging
            all_loss += loss
            all_ssim += ssim_score
            num_batches += 1
            writer.writerow([b_idx, f"{loss:.6f}", f"{ssim_score:.6f}"])
            pbar.set_postfix(mse=loss, ssim=ssim_score)

            # Save one grid from the first batch and then every 10th batch
            if b_idx % 10 == 0:
                grid_path = os.path.join(OUT_DIR, f"test_batch_{b_idx:04d}_grid.png")
                save_recon_grid(batch, recon, grid_path)

            if SAVE_PER_IMAGE:
                per_img_dir = os.path.join(OUT_DIR, "per_image")
                save_batch_images(batch, recon, per_img_dir, b_idx)

avg_loss = all_loss / max(1, num_batches)
avg_ssim = all_ssim / max(1, num_batches)

# Summary
print(f"\n[Test Summary] MSE: {avg_loss:.6f} | SSIM: {avg_ssim:.6f}")
with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
    f.write(f"Test size: {len(test_set)} images\n")
    f.write(f"Avg MSE: {avg_loss:.6f}\n")
    f.write(f"Avg SSIM: {avg_ssim:.6f}\n")
    f.write(f"Outputs saved to: {OUT_DIR}\n")

print(f"Predictions complete. Outputs saved in: {OUT_DIR}")
