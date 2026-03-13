import nbformat as nbf

nb = nbf.v4.new_notebook()

code1 = """# Cell 1 - Setup
import sys; sys.path.insert(0, '../..')
import torch, torch.nn as nn
from ml.training.model import MODEL_REGISTRY
from ml.training.dataloader import get_dataloaders
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)"""

code2 = """# Cell 2 - Load data
train_loader, val_loader, _ = get_dataloaders(batch_size=32, num_workers=0)"""

code3 = """# Cell 3 - Build model
model = MODEL_REGISTRY['densenet121'](pretrained=True, freeze_layers=True).to(device)"""

code4 = """# Cell 4 - Verify trainable params
total   = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total params: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.1f}%)')"""

code5 = """# Cell 5 - Run 1 epoch to verify everything works
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
model.train()
for i, (imgs, labels) in enumerate(train_loader):
    imgs, labels = imgs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(imgs)
    loss = criterion(outputs, labels)
    loss.backward(); optimizer.step()
    if i % 20 == 0: print(f'Batch {i}/{len(train_loader)} Loss: {loss.item():.4f}')
    if i == 60: break   # Just test 60 batches"""

nb['cells'] = [
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_code_cell(code5),
]

with open('ml/notebooks/03_transfer_learning.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Transfer learning notebook created successfully.")
