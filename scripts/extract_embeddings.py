# File: extract_embeddings.py (standalone)

import os
import torch
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
from transformers import AutoModel
from torchvision import transforms as T

# ======= Configs =======
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINO_MODEL_NAME = "facebook/dinov2-base"
MRI_SLICE_DIR = "/oak/stanford/groups/ogevaert/maxvpuyv/projects/brain/data/TCGA_set1_T1"
LABEL_SLICE_DIR = "/oak/stanford/groups/ogevaert/maxvpuyv/projects/brain/data/TCGA_set1_labels"
PROJ_HEAD_CKPT = "/oak/stanford/groups/ogevaert/maxvpuyv/projects/brain/student_head.pth"
OUTPUT_DIR = "tumor_embeddings_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======= Model Components =======
class ProjectionHead(torch.nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.GELU(),
            torch.nn.Linear(out_dim, out_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.proj(x)

# ======= Crop Utility =======
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

# ======= Load Model =======
print("Loading model...")
vit = AutoModel.from_pretrained(DINO_MODEL_NAME).to(DEVICE).eval()
head = ProjectionHead(vit.config.hidden_size).to(DEVICE).eval()
head.load_state_dict(torch.load(PROJ_HEAD_CKPT, map_location=torch.device(DEVICE)))


# ======= Dataset Setup =======
nii_paths = sorted(glob(os.path.join(MRI_SLICE_DIR, "*.nii.gz")))
label_paths = sorted(glob(os.path.join(LABEL_SLICE_DIR, "*.nii.gz")))

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((518, 518)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor()
])

features = []
filenames = []

# ======= Embedding Extraction =======
print("Extracting embeddings...")
for img_path, lbl_path in tqdm(zip(nii_paths, label_paths), total=len(nii_paths)):
    img = nib.load(img_path).get_fdata().astype(np.float32)
    lbl = nib.load(lbl_path).get_fdata().astype(np.uint8)
    img = img / (np.max(img) + 1e-8)

    cropped, _ = crop_to_fixed_tumor_box(img, lbl)
    if cropped is None:
        continue

    for i in range(cropped.shape[2]):
        slice_2d = np.uint8(255 * np.clip(cropped[:, :, i], 0, 1))
        if np.std(slice_2d) < 1e-5:
            continue

        slice_tensor = transform(slice_2d).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feat = vit(slice_tensor).last_hidden_state[:, 0, :]
            proj = head(feat).squeeze().cpu().numpy()
            features.append(proj)
            filenames.append(os.path.basename(img_path).replace(".nii.gz", f"_z{i}"))

# ======= Save Outputs =======
features = np.stack(features)
np.save(os.path.join(OUTPUT_DIR, "tumor_slice_embeddings.npy"), features)
with open(os.path.join(OUTPUT_DIR, "filenames.txt"), "w") as f:
    f.writelines([fn + "\n" for fn in filenames])

print("✅ Done. Embeddings saved to:", OUTPUT_DIR)
