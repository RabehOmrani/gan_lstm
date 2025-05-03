import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
import time

def train_gan(generator, discriminator, train_loader, val_loader, device, 
              noise_dim=100, n_epochs=15, lr=0.0002, beta1=0.5, beta2=0.999,
              save_interval=5, sample_interval=500, log_interval=100, 
              results_dir='results', models_dir='models'):
    """
    Train the LSTM-GAN model
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use for training
        noise_dim: Dimension of the noise vector
        n_epochs: Number of epochs
        lr: Learning rate
        beta1: Beta1 parameter for Adam optimizer
        beta2: Beta2 parameter for Adam optimizer
        save_interval: Interval for saving models
        sample_interval: Interval for generating samples
        log_interval: Interval for logging
        results_dir: Directory to save results
        models_dir: Directory to save models
        
    Returns:
        history: Dictionary with training history
    """
    # Create directories for results and models
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    classification_loss = nn.CrossEntropyLoss()
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    
    # Labels for real and fake
    real_label = 0.9  # Label smoothing
    fake_label = 0.0
    
    # Training history
    history = {
        'D_losses': [],
        'G_losses': [],
        'D_real_acc': [],
        'D_fake_acc': [],
        'D_class_acc': [],
        'val_losses': [],
        'val_acc': []
    }
    
    # Function to evaluate the discriminator on validation data
    def evaluate_discriminator(discriminator, val_loader):
        discriminator.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for signals, _, labels_idx in val_loader:
                signals = signals.to(device)
                labels_idx = labels_idx.to(device)
                
                # Forward pass
                validity, class_pred, _ = discriminator(signals)
                
                # Calculate classification accuracy
                _, predicted = torch.max(class_pred.data, 1)
                total += labels_idx.size(0)
                correct += (predicted == labels_idx).sum().item()
                
                # Calculate classification loss
                loss = classification_loss(class_pred, labels_idx)
                val_loss += loss.item()
        
        # Calculate average loss and accuracy
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        
        return val_loss, val_acc
    
    # Training loop
    print('Starting training...')
    start_time = time.time()
    
    for epoch in range(n_epochs):
        generator.train()
        discriminator.train()
        
        # Initialize metrics
        epoch_d_loss = 0
        epoch_g_loss = 0
        d_real_acc = 0
        d_fake_acc = 0
        d_class_acc = 0
        
        # Progress bar
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        
        for i, (real_ecgs, labels, labels_idx) in pbar:
            batch_size = real_ecgs.size(0)
            
            # Move data to device
            real_ecgs = real_ecgs.to(device)
            labels = labels.to(device)
            labels_idx = labels_idx.to(device)
            
            # Create labels for real and fake samples
            real_targets = torch.full((batch_size,), real_label, device=device)
            fake_targets = torch.full((batch_size,), fake_label, device=device)
            
            # -----------------
            # Train Discriminator
            # -----------------
            optimizer_D.zero_grad()
            
            # Real ECG
            real_validity, real_class_pred, _ = discriminator(real_ecgs)
            
            # Calculate loss for real samples
            d_real_loss = adversarial_loss(real_validity, real_targets)
            d_class_loss = classification_loss(real_class_pred, labels_idx)
            
            # Calculate accuracy for real samples
            _, real_pred = torch.max(real_class_pred, 1)
            d_class_acc += (real_pred == labels_idx).sum().item() / batch_size
            d_real_acc += ((real_validity > 0.5).float() == real_targets).float().mean().item()
            
            # Generate fake ECG
            z = torch.randn(batch_size, noise_dim, device=device)
            gen_ecgs = generator(z, labels)
            
            # Classify fake ECG
            fake_validity, fake_class_pred, _ = discriminator(gen_ecgs.detach())
            
            # Calculate loss for fake samples
            d_fake_loss = adversarial_loss(fake_validity, fake_targets)
            d_fake_class_loss = classification_loss(fake_class_pred, labels_idx)
            
            # Calculate accuracy for fake samples
            d_fake_acc += ((fake_validity < 0.5).float() == (1 - fake_targets)).float().mean().item()
            
            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss + d_class_loss + d_fake_class_loss
            
            # Backpropagation
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            # Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate fake ECG
            z = torch.randn(batch_size, noise_dim, device=device)
            gen_ecgs = generator(z, labels)
            
            # Classify fake ECG
            fake_validity, fake_class_pred, _ = discriminator(gen_ecgs)
            
            # Calculate generator loss
            g_adv_loss = adversarial_loss(fake_validity, real_targets)
            g_class_loss = classification_loss(fake_class_pred, labels_idx)
            g_loss = g_adv_loss + g_class_loss
            
            # Backpropagation
            g_loss.backward()
            optimizer_G.step()
            
            # Update metrics
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            
            # Update progress bar
            pbar.set_description(f'Epoch {epoch+1}/{n_epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}')
            
            # Generate and save samples at intervals
            batches_done = epoch * len(train_loader) + i
            if batches_done % sample_interval == 0:
                # Generate samples for each class
                n_classes = labels.size(1)
                for c in range(n_classes):
                    # Generate samples
                    generator.eval()
                    with torch.no_grad():
                        z = torch.randn(5, noise_dim, device=device)
                        gen_labels = torch.zeros(5, n_classes, device=device)
                        gen_labels[:, c] = 1.0
                        samples = generator(z, gen_labels)
                    generator.train()
                    
                    # Plot samples
                    plt.figure(figsize=(15, 10))
                    for j in range(5):
                        plt.subplot(5, 1, j+1)
                        plt.plot(samples[j, 0, :].cpu().numpy())  # Plot first lead
                        plt.title(f'Class: {c}')
                    plt.tight_layout()
                    plt.savefig(f'{results_dir}/epoch_{epoch+1}_batch_{batches_done}_class_{c}.png')
                    plt.close()
        
        # Calculate average metrics for the epoch
        epoch_d_loss /= len(train_loader)
        epoch_g_loss /= len(train_loader)
        d_real_acc /= len(train_loader)
        d_fake_acc /= len(train_loader)
        d_class_acc /= len(train_loader)
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate_discriminator(discriminator, val_loader)
        generator.train()  # Ensure the generator is back in training mode
        
        # Update history
        history['D_losses'].append(epoch_d_loss)
        history['G_losses'].append(epoch_g_loss)
        history['D_real_acc'].append(d_real_acc)
        history['D_fake_acc'].append(d_fake_acc)
        history['D_class_acc'].append(d_class_acc)
        history['val_losses'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{n_epochs} | D Loss: {epoch_d_loss:.4f} | G Loss: {epoch_g_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save models at intervals
        if (epoch + 1) % save_interval == 0:
            torch.save(generator.state_dict(), f'{models_dir}/generator_epoch_{epoch+1}.pt')
            torch.save(discriminator.state_dict(), f'{models_dir}/discriminator_epoch_{epoch+1}.pt')
            
            # Plot and save training history
            from src.utils.visualization import plot_training_history
            plot_training_history(history, save_path=f'{results_dir}/training_history_epoch_{epoch+1}.png')
    
    # Save final models
    torch.save(generator.state_dict(), f'{models_dir}/generator_final.pt')
    torch.save(discriminator.state_dict(), f'{models_dir}/discriminator_final.pt')
    
    # Save training history
    with open(f'{results_dir}/training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f'Training completed in {training_time/60:.2f} minutes')
    print('Final validation accuracy: {:.2f}%'.format(history['val_acc'][-1]))
    
    return history