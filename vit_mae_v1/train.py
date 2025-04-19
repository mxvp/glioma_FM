# train_mae.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from vit_model import ViTMAE
from dataset import TumorPatchDataset  # assumes you've set this up
import torch.optim as optim

# ---- CONFIG ----
IMAGE_DIR = "data/TCGA_set2_images"
LABEL_DIR = "data/TCGA_set2_labels"
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "mae_encoder.pt"
MASK_RATIO = 0.75

# ---- DATA ----
dataset = TumorPatchDataset(IMAGE_DIR, LABEL_DIR, patch_size=(224,192,160))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# ---- MODEL ----
model = ViTMAE(patch_size=16, volume_size=(224,192,160), in_channels=1).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# ---- TRAINING ----
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for x, _ in pbar:
        x = x.to(DEVICE)  # [B, 1, 64, 64, 64]

        optimizer.zero_grad()
        predicted, mask_idx, _ = model(x, mask_ratio=MASK_RATIO)
        target_patches = model.patchify(x)  # [B, N, 4096]


        # Compute loss over masked patches only
        B, N, D = predicted.shape
        batch_range = torch.arange(B)[:, None].to(DEVICE)
        masked_pred = predicted[batch_range, mask_idx]  # [B, M, D]
        masked_true = target_patches[batch_range, mask_idx]

        loss = loss_fn(masked_pred, masked_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} - Avg Loss: {total_loss / len(loader):.4f}")

    # Save model every few epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, SAVE_PATH)
        

print("done")