import os, pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
import warnings; warnings.filterwarnings('ignore')
 
DATA_ROOT = 'ml/data/raw'
 
splits = ['train', 'val', 'test']
classes = ['NORMAL', 'PNEUMONIA']
counts = {s: {c: len(list(pathlib.Path(f'{DATA_ROOT}/{s}/{c}').glob('*.jpeg')))
              for c in classes} for s in splits}
print(counts)
 
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, split in zip(axes, splits):
    ax.bar(classes, [counts[split][c] for c in classes],
           color=['#2563EB','#DC2626'])
    ax.set_title(f'{split.upper()} SET')
    ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('ml/reports/class_distribution.png')
print("Saved class_distribution.png")
 
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, cls in enumerate(classes):
    folder = pathlib.Path(f'{DATA_ROOT}/train/{cls}')
    imgs = list(folder.glob('*.jpeg'))[:4]
    for j, img_path in enumerate(imgs):
        ax = axes[i][j]
        ax.imshow(Image.open(img_path), cmap='gray')
        ax.set_title(f'{cls}'); ax.axis('off')
plt.tight_layout()
plt.savefig('ml/reports/sample_images.png')
print("Saved sample_images.png")
 
sizes = []
for img_path in pathlib.Path(f'{DATA_ROOT}/train/NORMAL').glob('*.jpeg'):
    sizes.append(Image.open(img_path).size)
widths, heights = zip(*sizes)
print(f'Width  - min:{min(widths)}, max:{max(widths)}, mean:{np.mean(widths):.0f}')
print(f'Height - min:{min(heights)}, max:{max(heights)}, mean:{np.mean(heights):.0f}')
