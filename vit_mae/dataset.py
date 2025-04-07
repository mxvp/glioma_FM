import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt

class TumorPatchDataset(Dataset):
    def __init__(self, image_dir, label_dir, patch_size=(224, 192, 160), transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.filenames = sorted(os.listdir(image_dir))
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def load_nifti(self, path):
        return nib.load(path).get_fdata().astype(np.float32)

    def extract_tumor_patch(self, image, mask):
        if image.ndim == 3:
            image = image[np.newaxis, ...]
    
        coords = np.argwhere(mask > 0)
        if coords.shape[0] == 0:
            raise ValueError("No tumor in mask")
    
        minc, maxc = coords.min(0), coords.max(0) + 1
        center = (minc + maxc) // 2
        patch = np.zeros((1, *self.patch_size), dtype=np.float32)
    
        for i, axis in enumerate(['x', 'y', 'z']):
            dim_size = image.shape[i + 1]
            half_size = self.patch_size[i] // 2
            start = max(center[i] - half_size, 0)
            end = min(start + self.patch_size[i], dim_size)
            start = max(end - self.patch_size[i], 0)  # adjust if near border
    
            if i == 0:
                x_slice = slice(start, end)
            elif i == 1:
                y_slice = slice(start, end)
            elif i == 2:
                z_slice = slice(start, end)
    
        cropped = image[:, x_slice, y_slice, z_slice]
    
        # Pad if needed (if at edge of brain)
        pad = [(0, 0)]  # for channel dim
        for i, size in enumerate(cropped.shape[1:]):
            total_pad = self.patch_size[i] - size
            before = total_pad // 2
            after = total_pad - before
            pad.append((before, after))
    
        cropped = np.pad(cropped, pad, mode='constant', constant_values=0)
    
        # Normalize
        mean, std = cropped.mean(), cropped.std() + 1e-5
        cropped = (cropped - mean) / std
    
        return cropped
    

    def __getitem__(self, idx):
        image_filename = self.filenames[idx]
        image = self.load_nifti(os.path.join(self.image_dir, image_filename))
        label_filename = image_filename.replace("_0000", "")
        label = self.load_nifti(os.path.join(self.label_dir, label_filename))

        patch = self.extract_tumor_patch(image, label)
        patch = torch.from_numpy(patch).float()

        if self.transform:
            patch = self.transform(patch)

        return patch, 0  # dummy label
