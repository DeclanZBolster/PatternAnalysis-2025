
"""
This file receives the data sets of train, validate, and test. 
The VQ-VAE structure adapts to training and intermittent validation
for effective reconstruction. Which is tested against the test data
set the model has not previously seen.
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

## Helper methods

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

# Main training loop
def main():
    

    ## Local data directories
    TRAIN_DIR = "C:/Users/s4699287/Desktop/A3_dataStorage/keras_slices_train"
    VAL_DIR = "C:/Users/s4699287/Desktop/A3_dataStorage/keras_slices_validate"
    TEST_DIR = "C:/Users/s4699287/Desktop/A3_dataStorage/keras_slices_test"

    ## Where training data is sent to.
    ## This was generated with AI
    OUT_DIR = os.path.join(
        "C:/Users/s4699287/Desktop/A3_localData/vqvae_outputs",
        datetime.now().strftime("%Y%m%d_%H%M"),
    )

    ## Local file location at which resulting data is sent to.
    ## line generated with AI.
    os.makedirs(OUT_DIR, exist_ok=True)

    ## Hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 4e-4
    IN_CHANNELS = 64
    EMBEDDING_DIM = 32 
    NUM_EMBEDDINGS = 128 
    OUT_CHANNELS = 1
    VAL_EVERY = 1
    ## This variable was generated with AI, as it is used the methods
    ## preiovusly mentioned also generated with AI
    ASSUME_MINMAX = True  ## Used since using Min-Max normalisation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ## Datasets and loaders
    train_set = MRI_dataset(path=TRAIN_DIR, earlyStop=False)
    val_set = MRI_dataset(path=VAL_DIR, earlyStop=False)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    ## Test set and loader
    test_set = MRI_dataset(path=TEST_DIR, earlyStop=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    print(f"Loaded {len(train_set)} training and {len(val_set)} validation images.")

    ## Model
    encoder = Encoder(IN_CHANNELS, EMBEDDING_DIM).to(device)
    decoder = Decoder(in_channel=EMBEDDING_DIM, out_channel=OUT_CHANNELS).to(device)
    vq_layer = VectorQuantiser(num_embeddings=NUM_EMBEDDINGS, embedding_dim=EMBEDDING_DIM).to(device)
    model = VQVAE(encoder, decoder, vq_layer).to(device)

    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Adding learn-rate scheduler
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimiser,
        mode="min",
        factor=0.5,
        patience=2,
        threshold=1e-4
    )
    ## Trackers
    train_losses, val_losses, train_ssims, val_ssims = [], [], [], []
    best_val_ssim = -1.0

    ## Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        ## Dropout layers remain active and batchNorm layers update
        ## their running estimates.
        model.train()
        epoch_loss, epoch_ssim = 0.0, 0.0

        current_lr = optimiser.param_groups[0]["lr"]

        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")


        print(f"learning rate: {current_lr:.6f}")
        
        ## This loading bar was generated from AI use
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")

        for batch in pbar:
            batch = batch.to(device)

            optimiser.zero_grad() ## Zeroing the gradient for training batch

            recon, vq_loss, _ = model(batch)
            recon_loss = criterion(recon, batch)
            loss = recon_loss + vq_loss

            loss.backward()
            ## This line was suggested from AI use
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) ## Limits the norm of the gradients to prevent gradient runaway
            optimiser.step()

            ## Calculation generated from AI use
            batch_ssim_val = batch_ssim(batch, recon, assume_minmax=ASSUME_MINMAX)

            epoch_loss += loss.item()
            epoch_ssim += batch_ssim_val
            # pbar.set_postfix(loss=loss.item(), ssim=batch_ssim_val)

            ## This loading bar of the different types of losses resulting from
            ## the model was suggested from AI use.
            pbar.set_postfix(
                loss=loss.item(),
                recon=recon_loss.item(),
                vq=vq_loss.item(),
                ssim=batch_ssim_val
            )

        avg_train_loss = epoch_loss / ( len(train_loader))
        
        avg_train_ssim = epoch_ssim / len(train_loader)
        
        train_losses.append(avg_train_loss)
        train_ssims.append(avg_train_ssim)
        print(f"[Train] Loss={avg_train_loss:.4f}, SSIM={avg_train_ssim:.4f}")

        ## Validation
        if epoch % VAL_EVERY == 0:
            ## Setting model to remove dropout layers and for
            ## the batchNorm laters to use a running mean instead
            model.eval()
            val_loss, val_ssim_total = 0.0, 0.0
            ## No gradient change because it is validation, not training
            with torch.no_grad():
                ## This line was generated from AI use
                for val_batch in tqdm(val_loader, desc="Validating", leave=False):

                    val_batch = val_batch.to(device)
                    val_recon, val_vq_loss, _ = model(val_batch)
                    vloss = criterion(val_recon, val_batch) + val_vq_loss
                    val_loss += vloss.item()
                    ## This line was generated from AI use
                    val_ssim_total += batch_ssim(val_batch, val_recon, assume_minmax=ASSUME_MINMAX)

                avg_val_loss = (val_loss / len(val_loader))
        
                avg_val_ssim = val_ssim_total / len(val_loader)
                val_losses.append(avg_val_loss)
                val_ssims.append(avg_val_ssim)

                print(f"[Val] Loss={avg_val_loss:.4f}, SSIM={avg_val_ssim:.4f}")

                ## Updating the LR scheduler
                scheduler.step(avg_val_loss)

                ## Saving reconstructions each epoch
                ## This block of code was generated by AI use
                save_path = os.path.join(OUT_DIR, f"epoch_{epoch:03d}_recons.png")
                save_recon_grid(val_batch, val_recon, save_path)
                print(f"Saved sample reconstructions to {save_path}")

                if avg_val_ssim > best_val_ssim:
                    best_val_ssim = avg_val_ssim

                    ## Saving best model to use in predict.py
                    torch.save(model.state_dict(), "model.path")

                    print(f"New best model saved (SSIM={best_val_ssim:.4f})")

    ## Save results of training and validation

    ## This block of code was generated by AI use
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend(); plt.xlabel("Epoch"); plt.title("Loss"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png")); plt.close()

    ## This block of code was generated by AI use
    plt.figure()
    plt.plot(train_ssims, label="Train SSIM")
    plt.plot(val_ssims, label="Val SSIM")
    plt.legend(); plt.xlabel("Epoch"); plt.title("SSIM"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ssim_curve.png")); plt.close()

    print(f"\nTraining complete. Outputs saved in: {OUT_DIR}")
    ##########################################################################
    ## Testing loop
    ##########################################################################
    
    ## Output directory of test results, if you later wish to
    ## This block of code was generated by AI use
    TEST_OUT_DIR = os.path.join(
    "C:/Users/s4699287/Desktop/A3_localData/vqvae_tests",
    datetime.now().strftime("%Y%m%d_%H%M"),
    )
    os.makedirs(TEST_OUT_DIR, exist_ok=True)

    ## Loading the best saved model from training and validation
    savedModel = torch.load("model.path", weights_only=True)
    model.load_state_dict(savedModel)

    model.eval()
    criterion = nn.MSELoss(reduction="mean")


    ## Loading test data
    test_set = MRI_dataset(path=TEST_DIR, earlyStop=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    print(f"Loaded {len(test_set)} test images.")

    ## Tracking results
    all_loss, all_ssim = 0.0, 0.0
    num_batches = 0
    lowestTestSSIM = 999
    highestTestSSIM = 0
    
    ## Setting no gradient changed because model
    ## should not change under testing
    with torch.no_grad():
        ## This line of code was generated by AI use
        pbar = tqdm(test_loader, desc="Testing")
        ## This block of code was generated by AI use
        for _, batch in enumerate(pbar):

            batch = batch.to(device)
            recon, _, _ = model(batch)

            loss = criterion(recon, batch).item()
            ssim_score = batch_ssim(batch, recon, assume_minmax=ASSUME_MINMAX)

            if ssim_score < lowestTestSSIM:
                lowestTestSSIM = ssim_score

            if ssim_score > highestTestSSIM:
                highestTestSSIM = ssim_score

            ## Logging SSIM and loss
            all_loss += loss
            all_ssim += ssim_score
            num_batches += 1
            pbar.set_postfix(mse=loss, ssim=ssim_score)


    avg_loss = all_loss / max(1, num_batches)
    avg_ssim = all_ssim / max(1, num_batches)

    # Summary
    print(f"\n[Test Summary] MSE: {avg_loss:.6f} | SSIM: {avg_ssim:.6f}")
    print(f"\nAvg SSIM: {avg_ssim:.6f}")
    print(f"\nLowest SSIM score: {lowestTestSSIM}")
    print(f"\nHighest SSIM score: {highestTestSSIM}")

    print(f"Testing complete")


## Running the train.py file
if __name__ == "__main__":
    main()