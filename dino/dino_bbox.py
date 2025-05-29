#!/usr/bin/env python

import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import nibabel as nib
from glob import glob
import warnings
from skimage.transform import resize
import json

# ========== Hardcoded dataset paths ==========
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

# ========== Pairing Strategies ==========
class DatasetPairingStrategy:
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def get_pairs(self):
        raise NotImplementedError

class TCGAPairingStrategy(DatasetPairingStrategy):
    def get_pairs(self):
        mask_files = sorted(glob(os.path.join(self.mask_dir, "*.nii*")))
        img_files = [
            os.path.join(self.img_dir, os.path.basename(f).replace(".nii", "_0000.nii"))
            for f in mask_files
        ]
        return list(zip(img_files, mask_files))

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
                pairs.append((image_file, mask_file))
            else:
                warnings.warn(f"Skipping case {case_folder_name}: missing T1c_bias or tumor_segmentation")
        return pairs

def get_strategy(name, img_dir, mask_dir):
    if name == "tcga":
        return TCGAPairingStrategy(img_dir, mask_dir)
    elif name == "pdgm":
        return PDGMPairingStrategy(img_dir, mask_dir)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def get_dataset_pairs(dataset_names):
    nii_paths, label_paths = [], []
    for name in dataset_names:
        paths = DATASET_PATHS[name]
        strategy = get_strategy(name, paths["img_dir"], paths["mask_dir"])
        pairs = strategy.get_pairs()
        nii_paths.extend([p[0] for p in pairs])
        label_paths.extend([p[1] for p in pairs])
    return nii_paths, label_paths

# ========== Configuration ==========
DINO_MODEL_NAME = "facebook/dinov2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== Dataset and Augmentations ==========
class MRISliceDataset(Dataset):
    def __init__(self, nii_paths, label_paths, crop_size=(80, 96), margin=10):
        self.pairs = list(zip(nii_paths, label_paths))
        self.slice_data = []
        self.crop_size = crop_size
        self.margin = margin if isinstance(margin, (list, tuple)) else [margin, margin]

        for img_path, lbl_path in self.pairs:
            img = nib.load(img_path).get_fdata().astype(np.float32)
            mask = nib.load(lbl_path).get_fdata().astype(np.uint8)
            img = img / (np.max(img) + 1e-8)

            # For each slice, crop to tumor bbox + margin, then resize
            for i in range(img.shape[2]):
                mask_slice = mask[:, :, i]
                if np.sum(mask_slice) == 0:
                    continue  # skip slices without tumor

                coords = np.array(np.where(mask_slice > 0))
                ymin, xmin = coords.min(axis=1)
                ymax, xmax = coords.max(axis=1)
                # Add margin, clip to image bounds
                ymin = max(0, ymin - self.margin[0])
                ymax = min(img.shape[0] - 1, ymax + self.margin[0])
                xmin = max(0, xmin - self.margin[1])
                xmax = min(img.shape[1] - 1, xmax + self.margin[1])

                img_crop = img[ymin:ymax+1, xmin:xmax+1, i]
                # Resize to crop_size (height, width)
                img_crop = resize(img_crop, self.crop_size, order=1, mode='constant', anti_aliasing=True)
                self.slice_data.append((img_crop, os.path.basename(img_path).replace(".nii.gz", f"_z{i}.npy")))

        self.aug1 = T.Compose([
            T.ToPILImage(),
            T.Resize((518, 518)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.03, 0.03)), 
            T.Grayscale(num_output_channels=3),
            T.ToTensor()
        ])
        self.aug2 = deepcopy(self.aug1)

    def __len__(self):
        return len(self.slice_data)

    def __getitem__(self, idx):
        slice_2d, fname = self.slice_data[idx]
        slice_2d = np.uint8(255 * np.clip(slice_2d, 0, 1))

        if np.std(slice_2d) < 1e-5:
            return self.__getitem__((idx + 1) % len(self))

        return self.aug1(slice_2d), self.aug2(slice_2d), fname

# ========== DINO Loss ==========
class DINOLoss(nn.Module):
    def __init__(self, out_dim, temp_student=0.2, temp_teacher=0.07, center_momentum=0.9):
        super().__init__()
        self.temp_student = temp_student
        self.temp_teacher = temp_teacher
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out):
        student_out = student_out / self.temp_student
        teacher_out = (teacher_out - self.center) / self.temp_teacher

        student_logprob = nn.functional.log_softmax(student_out, dim=-1)
        teacher_prob = nn.functional.softmax(teacher_out, dim=-1)

        loss = -torch.sum(teacher_prob * student_logprob, dim=-1).mean()
        self.update_center(teacher_out)
        return loss

    def update_center(self, teacher_out):
        batch_center = torch.mean(teacher_out, dim=0, keepdim=True)
        if torch.isnan(batch_center).any():
            print("NaNs in batch center — skipping update.")
            return
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

# ========== Projection Head ==========
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.proj(x)

# ========== Load Models ==========
def build_dino_vit():
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    vit = AutoModel.from_pretrained("facebook/dinov2-base")
    return vit.to(DEVICE), processor

# ========== Training ==========

def train(dataset_names, epochs, batch_size, save_ckpt_steps, num_workers, save_dir, crop_size, margin):
    vit, processor = build_dino_vit()
    teacher = deepcopy(vit).eval()
    student_head = ProjectionHead(vit.config.hidden_size).to(DEVICE)
    teacher_head = deepcopy(student_head).eval()
    loss_fn = DINOLoss(out_dim=256).to(DEVICE)

    opt = torch.optim.AdamW(student_head.parameters(), lr=1e-5, weight_decay=1e-2)

    nii_paths, label_paths = get_dataset_pairs(dataset_names)
    dataset = MRISliceDataset(nii_paths, label_paths, crop_size=crop_size, margin=margin)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    loss_log = []

    # Add loss log path
    loss_log_path = os.path.join(save_dir, "loss_log.json")

    global_step = 0
    for epoch in range(epochs):
        total_loss = 0
        running_loss = []
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for i, (view1, view2, fnames) in enumerate(pbar):
            view1 = view1.to(DEVICE)
            view2 = view2.to(DEVICE)

            with torch.no_grad():
                teacher_embeds = teacher(view2).last_hidden_state[:, 0, :]
                teacher_out = teacher_head(teacher_embeds)

            student_embeds = vit(view1).last_hidden_state[:, 0, :]
            student_out = student_head(student_embeds)

            if torch.isnan(student_out).any() or torch.isnan(teacher_out).any():
                print("NaNs detected in embeddings!")
                continue

            loss = loss_fn(student_out, teacher_out)
            if torch.isnan(loss):
                print("NaN loss! Skipping batch.")
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_head.parameters(), max_norm=1.0)
            opt.step()

            total_loss += loss.item()
            running_loss.append(loss.item())

            if len(running_loss) >= 20:
                avg_loss = sum(running_loss[-20:]) / 20
            else:
                avg_loss = sum(running_loss) / len(running_loss)

            pbar.set_postfix({"Rolling loss": f"{avg_loss:.4f}"})

            global_step += 1

            # Save checkpoint every save_ckpt_steps steps
            if save_ckpt_steps > 0 and global_step % save_ckpt_steps == 0:
                ckpt = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "backbone": vit.state_dict(),
                    "head": student_head.state_dict(),
                    "optimizer": opt.state_dict(),
                    "loss_log": loss_log,
                }
                ckpt_path = os.path.join(checkpoints_dir, f"checkpoint_step{global_step}_epoch{epoch+1}.pt")
                torch.save(ckpt, ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")

        mean_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} - Loss: {mean_loss:.4f}")
        loss_log.append({"epoch": epoch+1, "loss": mean_loss})

        # Write loss log after each epoch
        with open(loss_log_path, "w") as f:
            json.dump(loss_log, f, indent=2)

    # Save final checkpoint at the end of training
    ckpt = {
        "epoch": epochs,
        "step": global_step,
        "backbone": vit.state_dict(),
        "head": student_head.state_dict(),
        "optimizer": opt.state_dict(),
        "loss_log": loss_log,
    }
    ckpt_path = os.path.join(checkpoints_dir, f"checkpoint_final.pt")
    torch.save(ckpt, ckpt_path)
    print(f"Final checkpoint saved: {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs='+', required=True, choices=list(DATASET_PATHS.keys()))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-ckpt-steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="dino_results", help="Directory to save results (checkpoints)")
    parser.add_argument("--crop-size", nargs=2, type=int, default=[80, 96], help="Crop size for tumor bbox (height width)")
    parser.add_argument("--margin", nargs=2, type=int, default=[10, 10], help="Margin to add to bbox (height width)")
    args = parser.parse_args()
    train(
        args.datasets, args.epochs, args.batch_size, args.save_ckpt_steps,
        args.num_workers, args.save_dir, tuple(args.crop_size), list(args.margin)
    )
