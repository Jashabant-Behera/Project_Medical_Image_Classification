import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
 
class ChestXRayDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.class_names = ['NORMAL', 'PNEUMONIA']
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['filepath']).convert('RGB')
        label = int(row['label'])
        if self.transform:
            image = self.transform(image)
        return image, label
 
    def get_class_weights(self):
        '''Compute weights for WeightedRandomSampler to handle class imbalance'''
        counts = self.df['label'].value_counts().sort_index().values
        total = len(self.df)
        weights_per_class = total / (len(counts) * counts)
        sample_weights = [weights_per_class[int(l)] for l in self.df['label']]
        return torch.FloatTensor(sample_weights)
