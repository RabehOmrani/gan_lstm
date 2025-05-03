import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
import os

def plot_ecg_samples(signals, class_names=None, class_indices=None, title='ECG Samples', save_path=None):
    """
    Plot ECG samples
    
    Args:
        signals: ECG signals with shape (n_samples, n_leads, n_timesteps)
        class_names: List of class names
        class_indices: List of class indices for each sample
        title: Plot title
        save_path: If not None, save the plot to this path
    """
    n_samples = min(5, len(signals))
    
    plt.figure(figsize=(15, 10))
    for i in range(n_samples):
        plt.subplot(n_samples, 1, i+1)
        plt.plot(signals[i, 0, :])  # Plot first lead
        if class_names is not None and class_indices is not None:
            plt.title(f'Class: {class_names[class_indices[i]]}')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_generated_samples(generator, noise_dim, n_classes, class_names, device='cuda', save_dir='results'):
    """
    Generate and plot samples for each class
    
    Args:
        generator: Trained generator model
        noise_dim: Dimension of the noise vector
        n_classes: Number of classes
        class_names: List of class names
        device: Device to use for generation
        save_dir: Directory to save the plots
    """
    from src.utils.sampling import generate_samples
    
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    for c in range(n_classes):
        samples, _ = generate_samples(generator, noise_dim, n_classes, n_samples=5, fixed_class=c, device=device)
        
        # Plot samples
        for j in range(5):
            plt.subplot(n_classes, 5, c*5 + j + 1)
            plt.plot(samples[j, 0, :])  # Plot first lead
            if j == 0:
                plt.ylabel(class_names[c])
            plt.xticks([])
            if c == n_classes - 1:
                plt.xlabel('Time')
    
    plt.suptitle('Generated ECG Samples by Class')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'generated_samples.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
        save_path: If not None, save the plot to this path
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_roc_curves(y_true, y_probs, class_names, save_path=None):
    """
    Plot ROC curves for each class
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        class_names: List of class names
        save_path: If not None, save the plot to this path
    """
    plt.figure(figsize=(10, 8))
    
    for i in range(len(class_names)):
        # Convert to one-vs-rest for ROC
        binary_labels = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc='lower right')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_tsne_visualization(features, labels, class_names, save_path=None):
    """
    Visualize feature space using t-SNE
    
    Args:
        features: Feature vectors
        labels: Corresponding labels
        class_names: List of class names
        save_path: If not None, save the plot to this path
    """
    print('Computing t-SNE visualization...')
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 10))
    for i, label in enumerate(class_names):
        mask = labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], label=label, alpha=0.7)
    
    plt.legend()
    plt.title('t-SNE Visualization of ECG Features')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Args:
        history: Dictionary with training history
        save_path: If not None, save the plot to this path
    """
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history['D_losses'], label='D Loss')
    plt.plot(history['G_losses'], label='G Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator and Discriminator Losses')
    
    # Plot discriminator accuracies
    plt.subplot(2, 2, 2)
    plt.plot(history['D_real_acc'], label='D Real Acc')
    plt.plot(history['D_fake_acc'], label='D Fake Acc')
    plt.plot(history['D_class_acc'], label='D Class Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Discriminator Accuracies')
    
    # Plot validation metrics
    plt.subplot(2, 2, 3)
    plt.plot(history['val_losses'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Validation Loss')
    
    plt.subplot(2, 2, 4)
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_model_comparison(models, accuracies, f1_scores, save_path=None):
    """
    Plot model comparison
    
    Args:
        models: List of model names
        accuracies: List of accuracy scores
        f1_scores: List of F1 scores
        save_path: If not None, save the plot to this path
    """
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy')
    plt.bar(x + width/2, f1_scores, width, label='F1 Score')
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x, models)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()