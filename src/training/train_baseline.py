import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import pickle
import os
from tqdm import tqdm
import time

def train_and_evaluate(model, train_loader, val_loader, test_loader, model_name, n_epochs=15, 
                       lr=0.001, save_dir='baseline'):
    """
    Train and evaluate a baseline model
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        model_name: Name of the model
        n_epochs: Number of epochs
        lr: Learning rate
        save_dir: Directory to save results
        
    Returns:
        accuracy: Test accuracy
        f1: Test F1 score
        results: Dictionary with detailed results
    """
    # Create directory for results
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = next(model.parameters()).device
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'train_losses': [],
        'train_acc': [],
        'val_losses': [],
        'val_acc': []
    }
    
    # Best model tracking
    best_val_acc = 0
    best_model_state = None
    
    print(f'Training {model_name}...')
    start_time = time.time()
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Progress bar
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        
        for i, (signals, _, labels_idx) in pbar:
            signals = signals.to(device)
            labels_idx = labels_idx.to(device)
            
            # Forward pass
            outputs, _ = model(signals)
            loss = criterion(outputs, labels_idx)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels_idx.size(0)
            train_correct += (predicted == labels_idx).sum().item()
            
            # Update metrics
            train_loss += loss.item()
            
            # Update progress bar
            pbar.set_description(f'Epoch {epoch+1}/{n_epochs} | Loss: {loss.item():.4f}')
        
        # Calculate average training metrics
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for signals, _, labels_idx in val_loader:
                signals = signals.to(device)
                labels_idx = labels_idx.to(device)
                
                # Forward pass
                outputs, _ = model(signals)
                loss = criterion(outputs, labels_idx)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels_idx.size(0)
                val_correct += (predicted == labels_idx).sum().item()
                
                # Update metrics
                val_loss += loss.item()
        
        # Calculate average validation metrics
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update history
        history['train_losses'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_losses'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f'Training completed in {training_time/60:.2f} minutes')
    
    # Load best model for evaluation
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []
    all_features = []
    
    with torch.no_grad():
        for signals, _, labels_idx in test_loader:
            signals = signals.to(device)
            labels_idx = labels_idx.cpu().numpy()
            
            # Forward pass
            outputs, features = model(signals)
            
            # Get predictions and features
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            features = features.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels_idx)
            all_features.extend(features)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_features = np.array(all_features)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    
    # Print classification report
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds))
    
    # Plot confusion matrix
    from src.utils.visualization import plot_confusion_matrix
    plot_confusion_matrix(all_labels, all_preds, None, 
                         title=f'Confusion Matrix - {model_name}',
                         save_path=f'{save_dir}/{model_name}_confusion_matrix.png')
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train')
    plt.plot(history['val_losses'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_name} - Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title(f'{model_name} - Accuracy')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_training_history.png')
    plt.close()
    
    # Save results
    results = {
        'accuracy': accuracy,
        'f1': f1,
        'predictions': all_preds,
        'true_labels': all_labels,
        'features': all_features,
        'history': history,
        'training_time': training_time
    }
    
    with open(f'{save_dir}/{model_name}_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return accuracy, f1, results