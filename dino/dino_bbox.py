#!/usr/bin/env python

import os
import sys
import argparse
import torch
import torch.nn as nn
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel
from copy import deepcopy
from tqdm import tqdm
import random
import numpy as np
import nibabel as nib
from glob import glob
import torchvision.utils as vutils
from PIL import Image
import warnings

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
OUTPUT_EMBEDDINGS = "/oak/stanford/groups/ogevaert/maxvpuyv/projects/brain/output_embeddings"
DEBUG_SAVE_DIR = "/oak/stanford/groups/ogevaert/maxvpuyv/projects/brain/debug_slices"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)

# ========== Utility: Bounding box crop ==========
def crop_to_fixed_tumor_box(image, label, target_size=(128, 128, 64)):
    coords = np.argwhere(label > 0)
    if coords.shape[0] == 0:
        return None, None

    center = np.mean(coords, axis=0).astype(int)
    half_size = np.array(target_size) // 2
    
    start = np.maximum(center - half_size, 0)
    end = start + target_size

    for i in range(3):
        if end[i] > image.shape[i]:
            end[i] = image.shape[i]
            start[i] = max(0, end[i] - target_size[i])

    img_cropped = image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    return img_cropped, (start, end)

# ========== Dataset and Augmentations ==========
class MRISliceDataset(Dataset):
    def __init__(self, nii_paths, label_paths, target_size=(128, 128, 64)):
        self.pairs = list(zip(nii_paths, label_paths))
        self.slice_data = []

        for img_path, lbl_path in self.pairs:
            img = nib.load(img_path).get_fdata().astype(np.float32)
            lbl = nib.load(lbl_path).get_fdata().astype(np.uint8)

            img = img / (np.max(img) + 1e-8)
            cropped_img, box = crop_to_fixed_tumor_box(img, lbl, target_size)
            if cropped_img is None:
                continue

            for i in range(cropped_img.shape[2]):
                self.slice_data.append((cropped_img[:, :, i], os.path.basename(img_path).replace(".nii.gz", f"_z{i}.npy")))

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

        if random.random() < 0.01:
            img = Image.fromarray(slice_2d)
            img.save(os.path.join(DEBUG_SAVE_DIR, fname.replace('.npy', '_raw.png')))
            vutils.save_image(self.aug1(slice_2d), os.path.join(DEBUG_SAVE_DIR, fname.replace('.npy', '_aug1.png')))
            vutils.save_image(self.aug2(slice_2d), os.path.join(DEBUG_SAVE_DIR, fname.replace('.npy', '_aug2.png')))

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
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
    vit = AutoModel.from_pretrained(DINO_MODEL_NAME)
    return vit.to(DEVICE), processor

# ========== Training ==========
def train(dataset_names):
    vit, processor = build_dino_vit()
    teacher = deepcopy(vit).eval()
    student_head = ProjectionHead(vit.config.hidden_size).to(DEVICE)
    teacher_head = deepcopy(student_head).eval()
    loss_fn = DINOLoss(out_dim=256).to(DEVICE)

    opt = torch.optim.AdamW(student_head.parameters(), lr=1e-5, weight_decay=1e-2)

    nii_paths, label_paths = get_dataset_pairs(dataset_names)
    dataset = MRISliceDataset(nii_paths, label_paths)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    for epoch in range(20):
        total_loss = 0
        running_loss = []
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for view1, view2, fnames in pbar:
            view1 = view1.to(DEVICE)
            view2 = view2.to(DEVICE)

            with torch.no_grad():
                teacher_embeds = teacher(view2).last_hidden_state[:, 0, :]
                teacher_out = teacher_head(teacher_embeds)

            student_embeds = vit(view1).last_hidden_state[:, 0, :]
            student_out = student_head(student_embeds)

            if torch.isnan(student_out).any() or torch.isnan(teacher_out).any():
                print("NaNs detected in embeddings!")
                print("  teacher_out max:", teacher_out.max().item(), "min:", teacher_out.min().item(), "std:", teacher_out.std().item())
                print("  student_out max:", student_out.max().item(), "min:", student_out.min().item(), "std:", student_out.std().item())
                for i, fname in enumerate(fnames):
                    save_path = os.path.join(DEBUG_SAVE_DIR, f"{fname.replace('.npy', '')}_nan_slice_{epoch}.png")
                    vutils.save_image(view1[i].cpu(), save_path)
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

        print(f"Epoch {epoch+1} - Loss: {total_loss / len(loader):.4f}")

    torch.save(student_head.state_dict(), "student_head.pth")
    print("Finished training. Projection head saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs='+', required=True, choices=list(DATASET_PATHS.keys()))
    args = parser.parse_args()
    train(args.datasets)