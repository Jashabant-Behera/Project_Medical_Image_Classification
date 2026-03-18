import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from ml.training.dataset import ChestXRayDataset
from ml.training.augmentations import train_transforms, val_transforms
 
def get_dataloaders(data_dir: str = 'ml/data/processed', batch_size: int = 32,
                    num_workers: int = 4, use_sampler: bool = True):
    train_ds = ChestXRayDataset(f'{data_dir}/train.csv', transform=train_transforms)
    val_ds   = ChestXRayDataset(f'{data_dir}/val.csv',   transform=val_transforms)
    test_ds  = ChestXRayDataset(f'{data_dir}/test.csv',  transform=val_transforms)
 
    sampler = None
    shuffle_train = True
    if use_sampler:
        weights = train_ds.get_class_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle_train = False   # Cannot use shuffle=True with sampler
 
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              shuffle=shuffle_train, num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
