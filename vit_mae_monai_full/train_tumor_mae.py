#!/usr/bin/env python

"""
Self‑supervised full-brain MAE pre‑training (no bounding box, fixed spatial size)
--------------------------------------------------------------------------------
Usage:
    python train_tumor_mae.py --datasets tcga pdgm --out_dir OUTPUT_DIR
"""

import sys, os
import math, json, glob, argparse, warnings, random
import numpy as np, nibabel as nib, torch, torch.nn as nn
from tqdm import tqdm
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd,
    ResizeWithPadOrCropd, NormalizeIntensityd, Compose
)
from monai.data import CacheDataset, DataLoader
from monai.utils import set_determinism
import matplotlib.pyplot as plt

from models.mae_vit_monai import MaskedAutoEncoderViT

# ---------- Hardcoded dataset paths ----------
DATASET_PATHS = {
    "tcga": {
        "img_dir": "data/TCGA_set2_images",
        "mask_dir": "data/TCGA_set2_labels",
    },
    "pdgm": {
        "img_dir": "/oak/stanford/groups/ogevaert/data/brain_mri_tumor_project/UCSF-PDGM-v3",
        "mask_dir": None,
    },
}

# ---------- Pairing Strategies ----------
class DatasetPairingStrategy:
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def get_pairs(self):
        raise NotImplementedError

class TCGAPairingStrategy(DatasetPairingStrategy):
    def get_pairs(self):
        mask_files = sorted(glob.glob(os.path.join(self.mask_dir, "*.nii*")))
        img_files = [
            os.path.join(self.img_dir, os.path.basename(f).replace(".nii", "_0000.nii"))
            for f in mask_files
        ]
        return [{"image": i, "mask": m} for i, m in zip(img_files, mask_files)]

class PDGMPairingStrategy(DatasetPairingStrategy):
    def get_pairs(self):
        case_dirs = [os.path.join(self.img_dir, d) for d in os.listdir(self.img_dir)
                     if os.path.isdir(os.path.join(self.img_dir, d))]
        pairs = []
        for case in case_dirs:
            case_folder_name = os.path.basename(case)
            patient_id = case_folder_name.split("_")[0]
            image_file = os.path.join(case, f"{patient_id}_T1c_bias.nii.gz")
            mask_file = os.path.join(case, f"{patient_id}_tumor_segmentation.nii.gz")
            if os.path.exists(image_file) and os.path.exists(mask_file):
                pairs.append({"image": image_file, "mask": mask_file})
            else:
                print(f"[!] Skipping case {case_folder_name}: missing T1c_bias or tumor_segmentation")
        return pairs

def get_strategy(name, img_dir, mask_dir):
    if name == "tcga":
        return TCGAPairingStrategy(img_dir, mask_dir)
    elif name == "pdgm":
        return PDGMPairingStrategy(img_dir, mask_dir)
    else:
        raise ValueError(f"Unknown dataset: {name}")

# ---------- Patchify/Unpatchify and Visualization ----------
def patchify(imgs, patch_size):
    pz, py, px = patch_size
    B, C, D, H, W = imgs.shape
    imgs = imgs.reshape(B, C, D // pz, pz, H // py, py, W // px, px)\
               .permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    N = (D // pz) * (H // py) * (W // px)
    P = pz * py * px
    return imgs.view(B, N, P)

def unpatchify(patches, img_shape, patch_size):
    B, C, D, H, W = img_shape
    pz, py, px = patch_size
    nz, ny, nx = D // pz, H // py, W // px
    patches = patches.view(B, nz, ny, nx, pz, py, px)
    img = patches.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    img = img.view(B, 1, D, H, W)
    return img

def save_slices(tensor, out_dir, prefix, epoch, axis=2, max_slices=3):
    """Save a few slices from a 5D tensor (B, C, D, H, W) as PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    arr = tensor.detach().cpu().numpy()
    for b in range(min(arr.shape[0], max_slices)):
        if axis == 2:  # D
            idx = arr.shape[2] // 2
            img = arr[b, 0, idx, :, :]
        elif axis == 3:  # H
            idx = arr.shape[3] // 2
            img = arr[b, 0, :, idx, :]
        elif axis == 4:  # W
            idx = arr.shape[4] // 2
            img = arr[b, 0, :, :, idx]
        else:
            continue
        plt.imsave(os.path.join(out_dir, f"{prefix}_epoch{epoch}_b{b}.png"), img, cmap="gray")

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs='+', required=True, choices=["tcga", "pdgm"])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save_vis_every", type=int, default=10, help="Save visualizations every N epochs")
    parser.add_argument("--spatial_size", nargs=3, type=int, default=[160,192,160], help="Spatial size for all images (D H W)")
    parser.add_argument("--save_ckpt_every", type=int, default=50, help="Save checkpoint every N epochs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_determinism(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- Get image/mask pairs
    data_dicts = []
    for name in args.datasets:
        paths = DATASET_PATHS[name]
        strategy = get_strategy(name, paths["img_dir"], paths["mask_dir"])
        pairs = strategy.get_pairs()
        data_dicts.extend(pairs)
    print(f"Total paired samples: {len(data_dicts)}")

    # ---------- Set spatial size manually
    spatial_size = args.spatial_size
    print("Fixed spatial_size:", spatial_size)

    # ---------- Transforms & loader
    transforms = Compose([
        LoadImaged(["image", "mask"]),
        EnsureChannelFirstd(["image", "mask"]),
        ResizeWithPadOrCropd(["image", "mask"], spatial_size=spatial_size),
        NormalizeIntensityd("image", nonzero=True, channel_wise=True),
    ])

    ds = CacheDataset(data=data_dicts, transform=transforms, cache_rate=1.0)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    print(f"Dataset: {len(ds)} samples | Batch: {args.batch} | Steps/Epoch: {len(loader)}")

    # ---------- Model & optimizer
    model = MaskedAutoEncoderViT(
        in_channels=1, img_size=tuple(spatial_size), patch_size=(16,16,16),
        hidden_size=1152, mlp_dim=4608, num_layers=12, num_heads=16,
        masking_ratio=0.30, decoder_hidden_size=1152,
        decoder_mlp_dim=4608, decoder_num_layers=6, decoder_num_heads=16,
        spatial_dims=3).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    recon_loss = nn.L1Loss()

    start_epoch, loss_log = 1, []
    log_path = os.path.join(args.out_dir, "loss_log.json")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optimizer'])
        sched.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        loss_log = checkpoint.get('loss_log', [])
        print(f"Resumed from checkpoint: {args.resume} (epoch {start_epoch - 1})")

    vis_dir = os.path.join(args.out_dir, "visualizations")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = []
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for i, batch in enumerate(pbar):
            x = batch["image"].to(device)
            pred, mask = model(x)
            target = patchify(x, model.patch_size)
            loss = recon_loss(pred[mask.bool()], target[mask.bool()])

            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

            running.append(loss.item())
            pbar.set_postfix(loss=loss.item())

            # Save example slices and predictions every save_vis_every epochs, pick a random image from the batch
            if epoch % args.save_vis_every == 0 and i == 0:
                b_idx = random.randint(0, x.shape[0] - 1)
                save_slices(x[b_idx:b_idx+1], vis_dir, "input", epoch)
                B, N, P = pred.shape
                patch_size = model.patch_size
                recon = torch.zeros_like(target)
                recon[mask.bool()] = pred[mask.bool()]
                recon[~mask.bool()] = target[~mask.bool()]
                recon_img = unpatchify(recon, x.shape, patch_size)
                save_slices(recon_img[b_idx:b_idx+1], vis_dir, "recon", epoch)
                target_img = unpatchify(target, x.shape, patch_size)
                save_slices(target_img[b_idx:b_idx+1], vis_dir, "target", epoch)

        sched.step()
        mean_loss = sum(running) / len(running)
        loss_log.append({"epoch": epoch, "loss": mean_loss})
        print(f"Epoch {epoch} complete | Mean Loss: {mean_loss:.4f}")

        with open(log_path, "w") as f:
            json.dump(loss_log, f, indent=2)

        if epoch % args.save_ckpt_every == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "scheduler": sched.state_dict(),
                "loss_log": loss_log,
            }
            path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch}.pt")
            torch.save(ckpt, path)
            print(f"Checkpoint saved: {path}")

if __name__ == "__main__":
    main()