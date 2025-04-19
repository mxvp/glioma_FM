#!/usr/bin/env python


# ============================ train_tumor_mae.py ============================
"""
Self‑supervised tumour‑only MAE pre‑training
-------------------------------------------
Usage:
    python train_tumor_mae.py \
        --img_dir data/TCGA_set1_T1 \
        --mask_dir data/TCGA_set1_labels \
        --out_dir runs/mae_tcga \
        --epochs 800 --batch 4
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math, json, glob, argparse, warnings
import numpy as np, nibabel as nib, torch, torch.nn as nn
from tqdm import tqdm
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, CropForegroundd,
    ResizeWithPadOrCropd, NormalizeIntensityd, Compose
)
from monai.data import CacheDataset, DataLoader
from vit_mae_v2.models.mae_vit_monai import MaskedAutoEncoderViT
from monai.utils import set_determinism

def bbox(mask):
    idx = np.where(mask > 0)
    return [idx[0].min(), idx[0].max(), idx[1].min(), idx[1].max(), idx[2].min(), idx[2].max()]

def compute_global_crop(mask_files, patch=16):
    max_dims = np.zeros(3, dtype=int)
    meta = {}
    for mfile in tqdm(mask_files, desc="scanning masks"):
        m = nib.load(mfile).get_fdata()
        if m.max() == 0:
            warnings.warn(f"No tumour found in {mfile}")
            continue
        bb = bbox(m)
        dims = [int(bb[1]-bb[0]+1), int(bb[3]-bb[2]+1), int(bb[5]-bb[4]+1)]
        max_dims = np.maximum(max_dims, dims)
        meta[os.path.basename(mfile)] = {"bbox": [int(x) for x in bb], "dims": dims}
    spatial_size = [int(math.ceil(d / patch) * patch) for d in max_dims]
    return spatial_size, meta

def patchify(imgs, patch_size):
    pz, py, px = patch_size
    B, C, D, H, W = imgs.shape
    imgs = imgs.reshape(B, C, D // pz, pz, H // py, py, W // px, px)\
               .permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    N = (D // pz) * (H // py) * (W // px)
    P = pz * py * px
    return imgs.view(B, N, P)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_determinism(seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- file paths
    mask_files = sorted(glob.glob(os.path.join(args.mask_dir, "*.nii*")))
    img_files = [os.path.join(args.img_dir, os.path.basename(f).replace(".nii", "_0000.nii")) for f in mask_files]
    data_dicts = [{"image": i, "mask": m} for i, m in zip(img_files, mask_files)]
    print(f"Found {len(data_dicts)} image-mask pairs")

    # ---------- crop size
    crop_json = os.path.join(args.out_dir, "crop_meta.json")
    if os.path.exists(crop_json):
        with open(crop_json) as fp:
            j = json.load(fp)
        spatial_size = j["spatial_size"]
    else:
        spatial_size, meta = compute_global_crop(mask_files)
        json.dump({"spatial_size": spatial_size, "meta": meta}, open(crop_json, "w"))
    print("Fixed spatial_size:", spatial_size)

    # ---------- transforms & loader
    transforms = Compose([
        LoadImaged(["image", "mask"]),
        EnsureChannelFirstd(["image", "mask"]),
        CropForegroundd(["image", "mask"], source_key="mask", margin=0),
        ResizeWithPadOrCropd(["image", "mask"], spatial_size=spatial_size),
        NormalizeIntensityd("image", nonzero=True, channel_wise=True),
    ])

    ds = CacheDataset(data=data_dicts, transform=transforms, cache_rate=1.0)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)

    print(f"Dataset: {len(ds)} samples | Batch: {args.batch} | Steps/Epoch: {len(loader)}")

    # ---------- model & optimizer
    model = MaskedAutoEncoderViT(
        in_channels=1, img_size=tuple(spatial_size), patch_size=(16,16,16),
        hidden_size=1152, mlp_dim=4608, num_layers=12, num_heads=16,
        masking_ratio=0.40, decoder_hidden_size=1152,
        decoder_mlp_dim=4608, decoder_num_layers=6, decoder_num_heads=16,
        spatial_dims=3).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    recon_loss = nn.L1Loss()

    start_epoch = 1
    log_path = os.path.join(args.out_dir, "loss_log.json")
    loss_log = []

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optimizer'])
        sched.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        loss_log = checkpoint.get('loss_log', [])
        print(f"Resumed from checkpoint: {args.resume} (epoch {start_epoch - 1})")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = []
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            x = batch["image"].to(device)
            pred, mask = model(x)
            target = patchify(x, model.patch_size)
            loss = recon_loss(pred[mask.bool()], target[mask.bool()])

            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

            running.append(loss.item())
            pbar.set_postfix(loss=loss.item())

        sched.step()
        mean_loss = sum(running) / len(running)
        loss_log.append({"epoch": epoch, "loss": mean_loss})

        print(f"Epoch {epoch} complete | Mean Loss: {mean_loss:.4f}")
        with open(log_path, "w") as f:
            json.dump(loss_log, f, indent=2)

        if epoch % 100 == 0 or epoch == args.epochs:
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