import nbformat as nbf

nb = nbf.v4.new_notebook()
text = """# Cell 1 - Test Data Pipeline"""
code = """import sys; sys.path.insert(0, '../..')
from ml.training.dataloader import get_dataloaders
 
# Added num_workers=0 to avoid multiprocessing issues in Windows notebooks without __main__ guards
train_loader, val_loader, test_loader = get_dataloaders(batch_size=8, num_workers=0)
images, labels = next(iter(train_loader))
print('Image batch shape:', images.shape)   # Expected: torch.Size([8, 3, 224, 224])
print('Label batch:', labels)               # Expected: tensor of 0s and 1s
print('Train batches:', len(train_loader))
print('Val batches:  ', len(val_loader))
print('Test batches: ', len(test_loader))"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(code)
]

with open('ml/notebooks/02_data_pipeline_test.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Pipeline notebook created successfully.")
