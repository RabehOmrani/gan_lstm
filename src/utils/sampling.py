import torch
import numpy as np

def generate_samples(generator, noise_dim, n_classes, n_samples=10, fixed_class=None, device='cuda'):
    """
    Generate synthetic ECG samples using the trained generator
    
    Args:
        generator: Trained generator model
        noise_dim: Dimension of the noise vector
        n_classes: Number of classes
        n_samples: Number of samples to generate
        fixed_class: If not None, generate samples for this specific class
        device: Device to use for generation
        
    Returns:
        gen_ecgs: Generated ECG signals
        labels: Corresponding labels
    """
    generator.eval()
    with torch.no_grad():
        # Generate random noise
        z = torch.randn(n_samples, noise_dim).to(device)
        
        # Generate labels (either fixed or random)
        if fixed_class is not None:
            # Create one-hot encoded labels for the specified class
            labels = torch.zeros(n_samples, n_classes).to(device)
            labels[:, fixed_class] = 1.0
        else:
            # Generate random class labels
            random_classes = torch.randint(0, n_classes, (n_samples,))
            labels = torch.zeros(n_samples, n_classes).to(device)
            for i, c in enumerate(random_classes):
                labels[i, c] = 1.0
        
        # Generate ECG signals
        gen_ecgs = generator(z, labels)
    
    return gen_ecgs.cpu().numpy(), labels.cpu().numpy()

def augment_batch(signals, labels, noise_factor=0.05):
    """
    Apply data augmentation to a batch of ECG signals
    
    Args:
        signals: Batch of ECG signals
        labels: Corresponding labels
        noise_factor: Amount of noise to add
        
    Returns:
        augmented_signals: Augmented signals
        augmented_labels: Corresponding labels
    """
    # Add random noise
    noise = torch.randn_like(signals) * noise_factor
    augmented_signals = signals + noise
    
    # Time warping (stretch/compress slightly)
    stretch_factor = torch.rand(signals.size(0), 1, 1) * 0.2 + 0.9  # 0.9-1.1
    time_indices = torch.arange(signals.size(2)).float().unsqueeze(0).unsqueeze(0)
    time_indices = time_indices.repeat(signals.size(0), 1, 1)
    time_indices = time_indices * stretch_factor
    time_indices = time_indices.long().clamp(0, signals.size(2) - 1)
    
    warped_signals = torch.zeros_like(signals)
    for i in range(signals.size(0)):
        for j in range(signals.size(1)):
            warped_signals[i, j] = signals[i, j, time_indices[i, 0]]
    
    return torch.cat([signals, augmented_signals, warped_signals]), torch.cat([labels, labels, labels])

class AugmentedDataLoader:
    """
    Data loader that applies augmentation to each batch
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
    
    def __iter__(self):
        self.iterator = iter(self.dataloader)
        return self
    
    def __next__(self):
        try:
            signals, labels, labels_idx = next(self.iterator)
            aug_signals, aug_labels = augment_batch(signals, labels)
            aug_labels_idx = torch.argmax(aug_labels, dim=1)
            return aug_signals, aug_labels, aug_labels_idx
        except StopIteration:
            raise StopIteration
    
    def __len__(self):
        return len(self.dataloader)