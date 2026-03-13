# Phase 2: Data Pipeline — Dataset, Augmentations & DataLoaders

## Objective
Convert raw categorical sets output in Phase 1 directly into dynamic tensors via Python class overriding, implementing ImageNet transformation standardizations alongside `torch.utils.data.WeightedRandomSampler`.

## Files Created
1. `ml/training/dataset.py` - Extends `torch.utils.data.Dataset` mapping CSV locators to `PIL.Image.open().convert('RGB')` outputs, and defines `get_class_weights()` calculating the inverse proportion of Normal/Pneumonia samples sequentially.
2. `ml/training/augmentations.py` - Standardizes parameters (`IMG_SIZE = 224`) relying centrally on `torchvision.transforms.Compose`. Employs `RandomHorizontalFlip(0.5)`, `RandomRotation(15)`, and `ColorJitter` uniquely for the training stream to counter overfitting tendencies. Test arrays operate strictly on Base Resize matrices.
3. `ml/training/dataloader.py` - PyTorch `DataLoader` encapsulations with batch configurations overriding native shuffling internally whenever the `WeightedRandomSampler` is instantiated.
4. `ml/notebooks/02_data_pipeline_test.ipynb` - Smoke tests the Dataloaders confirming an image output `torch.Size([8, 3, 224, 224])` structurally.

## Methodologies
Class imbalance was explicitly mitigated. A raw training run on the native dataset structurally forces the Convolutional Neural Net into bias toward "PNEUMONIA" guessing due to 70%+ occurrences. The custom dataset dynamic sampler artificially corrects this balance probabilistically during DataLoader pulls without modifying files directly on disk.
