from utils.dataset import TumorPatchDataset
import matplotlib.pyplot as plt
import numpy as mp

dataset = TumorPatchDataset("data/TCGA_set2_images", "data/TCGA_set2_labels", patch_size=(224,192,160))

# pick random sample
patch, _ = dataset[0]  # shape [1, 224, 192, 160]
patch_np = patch.numpy()[0]

# plot central slices
fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs[0].imshow(patch_np[patch_np.shape[0]//2,:,:], cmap='gray')
axs[0].set_title('Axial (XY)')
axs[1].imshow(patch_np[:,patch_np.shape[1]//2,:], cmap='gray')
axs[1].set_title('Coronal (XZ)')
axs[2].imshow(patch_np[:,:,patch_np.shape[2]//2], cmap='gray')
axs[2].set_title('Sagittal (YZ)')
plt.show()
