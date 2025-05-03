import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import pickle
import os

def evaluate_discriminator(discriminator, test_loader, class_names, device='cuda', output_dir='evaluation'):
    """
    Evaluate the discriminator on the test set
    
    Args:
        discriminator: Trained discriminator model
        test_loader: Test data loader
        class_names: List of class names
        device: Device to use for evaluation
        output_dir: Directory to save evaluation results
        
    Returns:
        accuracy: Test accuracy
        f1: Test F1 score
        all_preds: Predicted labels
        all_labels: True labels
        all_probs: Predicted probabilities
        all_features: Feature vectors
    """
    os.makedirs(output_dir, exist_ok=True)
    
    discriminator.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []
    
    with torch.no_grad():
        for signals, _, labels_idx in test_loader:
            signals = signals.to(device)
            labels_idx = labels_idx.cpu().numpy()
            
            # Forward pass
            validity, class_pred, features = discriminator(signals)
            
            # Get predictions and probabilities
            probs = torch.softmax(class_pred, dim=1).cpu().numpy()
            preds = torch.argmax(class_pred, dim=1).cpu().numpy()
            features = features.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels_idx)
            all_probs.extend(probs)
            all_features.extend(features)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_features = np.array(all_features)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    
    # Print classification report
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Plot confusion matrix
    from src.utils.visualization import plot_confusion_matrix
    plot_confusion_matrix(all_labels, all_preds, class_names, 
                         save_path=f'{output_dir}/confusion_matrix.png')
    
    # Plot ROC curves
    from src.utils.visualization import plot_roc_curves
    plot_roc_curves(all_labels, all_probs, class_names, 
                   save_path=f'{output_dir}/roc_curves.png')
    
    # Visualize feature space using t-SNE
    from src.utils.visualization import plot_tsne_visualization
    plot_tsne_visualization(all_features, all_labels, class_names, 
                           save_path=f'{output_dir}/tsne_visualization.png')
    
    # Save evaluation results
    evaluation_results = {
        'accuracy': accuracy,
        'f1': f1,
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs,
        'features': all_features
    }
    
    with open(f'{output_dir}/evaluation_results.pkl', 'wb') as f:
        pickle.dump(evaluation_results, f)
    
    return accuracy, f1, all_preds, all_labels, all_probs, all_features

def compare_models(gan_results_path, lstm_results_path, lstm_aug_results_path, class_names, output_dir='evaluation'):
    """
    Compare the performance of different models
    
    Args:
        gan_results_path: Path to GAN evaluation results
        lstm_results_path: Path to LSTM baseline results
        lstm_aug_results_path: Path to LSTM+AugGAN results
        class_names: List of class names
        output_dir: Directory to save comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    with open(gan_results_path, 'rb') as f:
        gan_results = pickle.load(f)
    
    with open(lstm_results_path, 'rb') as f:
        lstm_results = pickle.load(f)
    
    with open(lstm_aug_results_path, 'rb') as f:
        lstm_aug_results = pickle.load(f)
    
    # Extract metrics
    gan_acc = gan_results['accuracy']
    gan_f1 = gan_results['f1']
    
    lstm_acc = lstm_results['accuracy']
    lstm_f1 = lstm_results['f1']
    
    lstm_aug_acc = lstm_aug_results['accuracy']
    lstm_aug_f1 = lstm_aug_results['f1']
    
    # Create comparison table
    models = ['LSTM', 'LSTM+AugGAN', 'LSTM-GAN (Ours)']
    accuracies = [lstm_acc, lstm_aug_acc, gan_acc]
    f1_scores = [lstm_f1, lstm_aug_f1, gan_f1]
    
    # Plot comparison
    from src.utils.visualization import plot_model_comparison
    plot_model_comparison(models, accuracies, f1_scores, save_path=f'{output_dir}/model_comparison.png')
    
    # Print comparison
    print('\nModel Comparison:')
    print(f'LSTM Classifier: Accuracy = {lstm_acc:.4f}, F1 = {lstm_f1:.4f}')
    print(f'LSTM+AugGAN: Accuracy = {lstm_aug_acc:.4f}, F1 = {lstm_aug_f1:.4f}')
    print(f'LSTM-GAN (Ours): Accuracy = {gan_acc:.4f}, F1 = {gan_f1:.4f}')
    
    # Calculate improvement
    acc_improvement = (gan_acc - lstm_acc) / lstm_acc * 100
    f1_improvement = (gan_f1 - lstm_f1) / lstm_f1 * 100
    
    print(f'\nImprovement over LSTM: Accuracy +{acc_improvement:.2f}%, F1 +{f1_improvement:.2f}%')
    
    # Calculate class-wise F1 scores
    from sklearn.metrics import f1_score
    
    # LSTM class-wise F1
    lstm_class_f1 = []
    for i in range(len(class_names)):
        true_binary = (lstm_results['true_labels'] == i)
        pred_binary = (lstm_results['predictions'] == i)
        f1 = f1_score(true_binary, pred_binary)
        lstm_class_f1.append(f1)
    
    # LSTM+AugGAN class-wise F1
    lstm_aug_class_f1 = []
    for i in range(len(class_names)):
        true_binary = (lstm_aug_results['true_labels'] == i)
        pred_binary = (lstm_aug_results['predictions'] == i)
        f1 = f1_score(true_binary, pred_binary)
        lstm_aug_class_f1.append(f1)
    
    # LSTM-GAN class-wise F1
    gan_class_f1 = []
    for i in range(len(class_names)):
        true_binary = (gan_results['true_labels'] == i)
        pred_binary = (gan_results['predictions'] == i)
        f1 = f1_score(true_binary, pred_binary)
        gan_class_f1.append(f1)
    
    # Plot class-wise F1 scores
    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.bar(x - width, lstm_class_f1, width, label='LSTM')
    plt.bar(x, lstm_aug_class_f1, width, label='LSTM+AugGAN')
    plt.bar(x + width, gan_class_f1, width, label='LSTM-GAN')
    
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('Class-wise F1 Score')
    plt.xticks(x, class_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/class_wise_f1.png')
    plt.close()
    
    # Save comparison results
    comparison_results = {
        'models': models,
        'accuracies': accuracies,
        'f1_scores': f1_scores,
        'lstm_class_f1': lstm_class_f1,
        'lstm_aug_class_f1': lstm_aug_class_f1,
        'gan_class_f1': gan_class_f1,
        'acc_improvement': acc_improvement,
        'f1_improvement': f1_improvement
    }
    
    with open(f'{output_dir}/comparison_results.pkl', 'wb') as f:
        pickle.dump(comparison_results, f)