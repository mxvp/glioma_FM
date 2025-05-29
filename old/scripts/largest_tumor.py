import os
import nibabel as nib
import numpy as np

label_dir = "data/TCGA_set2_labels"

max_dims = [0, 0, 0]

for fname in sorted(os.listdir(label_dir)):
    mask = nib.load(os.path.join(label_dir, fname)).get_fdata()
    coords = np.argwhere(mask > 0)
    if coords.shape[0] == 0:
        continue
    minc, maxc = coords.min(0), coords.max(0)
    dims = maxc - minc + 1
    max_dims = np.maximum(max_dims, dims)

print("Largest tumor bounding box (xyz):", max_dims)
