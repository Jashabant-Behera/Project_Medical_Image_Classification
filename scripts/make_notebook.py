import nbformat as nbf

nb = nbf.v4.new_notebook()
text = """# Cell 1 - Imports"""
code1 = """import os, pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
import warnings; warnings.filterwarnings('ignore')

DATA_ROOT = '../../data/raw'"""

code2 = """# Cell 2 - Count images per split and class
splits = ['train', 'val', 'test']
classes = ['NORMAL', 'PNEUMONIA']
counts = {s: {c: len(list(pathlib.Path(f'{DATA_ROOT}/{s}/{c}').glob('*.jpeg')))
              for c in classes} for s in splits}
print(counts)"""

code3 = """# Cell 3 - Plot class distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, split in zip(axes, splits):
    ax.bar(classes, [counts[split][c] for c in classes],
           color=['#2563EB','#DC2626'])
    ax.set_title(f'{split.upper()} SET')
    ax.set_ylabel('Count')
plt.tight_layout(); plt.savefig('../../reports/class_distribution.png')
plt.show()"""

code4 = """# Cell 4 - Display sample images
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, cls in enumerate(classes):
    folder = pathlib.Path(f'{DATA_ROOT}/train/{cls}')
    imgs = list(folder.glob('*.jpeg'))[:4]
    for j, img_path in enumerate(imgs):
        ax = axes[i][j]
        ax.imshow(Image.open(img_path), cmap='gray')
        ax.set_title(f'{cls}'); ax.axis('off')
plt.tight_layout(); plt.savefig('../../reports/sample_images.png')
plt.show()"""

code5 = """# Cell 5 - Image size distribution
sizes = []
for img_path in pathlib.Path(f'{DATA_ROOT}/train/NORMAL').glob('*.jpeg'):
    sizes.append(Image.open(img_path).size)
widths, heights = zip(*sizes)
print(f'Width  - min:{min(widths)}, max:{max(widths)}, mean:{np.mean(widths):.0f}')
print(f'Height - min:{min(heights)}, max:{max(heights)}, mean:{np.mean(heights):.0f}')"""

nb['cells'] = [
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_code_cell(code5),
]

with open('ml/notebooks/01_eda.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created successfully.")
