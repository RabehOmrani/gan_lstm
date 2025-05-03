import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        """
        Initialize ECG dataset
        
        Args:
            signals: ECG signals with shape (n_samples, n_leads, n_timesteps)
            labels: One-hot encoded labels with shape (n_samples, n_classes)
        """
        self.signals = torch.tensor(signals, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.label_indices = torch.tensor(np.argmax(labels, axis=1), dtype=torch.long)
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx], self.label_indices[idx]