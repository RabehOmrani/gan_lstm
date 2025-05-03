import torch
import argparse
import os
import pickle

def main():
    parser = argparse.ArgumentParser(description='ECG-LSTM-GAN: Generative Adversarial LSTM for End-to-End ECG Diagnosis')
    parser.add_argument('--data_path', type=str, default='/path/to/ptb-xl-dataset/', help='Path to the PTB-XL dataset')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--mode', type=str, choices=['preprocess', 'train_gan', 'train_baseline', 'evaluate', 'all'], default='all', help='Mode of operation')
    parser.add_argument('--sampling_rate', type=int, default=100, help='Sampling rate (100 or 500 Hz)')
    parser.add_argument('--noise_dim', type=int, default=100, help='Dimension of the noise vector')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension of LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 parameter for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 parameter for Adam optimizer')
    parser.add_argument('--save_interval', type=int, default=5, help='Interval for saving models')
    parser.add_argument('--sample_interval', type=int, default=500, help='Interval for generating samples')
    parser.add_argument('--log_interval', type=int, default=100, help='Interval for logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_oversampling', action='store_true', help='Use oversampling for data balancing')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (-1 for CPU)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    print(f'Using device: {device}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Preprocess data
    if args.mode in ['preprocess', 'all']:
        print('Preprocessing data...')
        from src.data.preprocessing import load_ptbxl_data, apply_oversampling, save_datasets
        
        # Load and preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test, metadata = load_ptbxl_data(
            args.data_path, args.sampling_rate
        )
        
        # Apply oversampling if specified
        if args.use_oversampling:
            X_train, y_train = apply_oversampling(X_train, np.argmax(y_train, axis=1))
        
        # Save datasets
        data_dir = os.path.join(args.output_dir, 'data')
        save_datasets(X_train, X_val, X_test, y_train, y_val, y_test, metadata, data_dir)
    
    # Load metadata
    try:
        with open(os.path.join(args.output_dir, 'data', 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        n_leads = metadata['n_leads']
        n_timesteps = metadata['n_timesteps']
        n_classes = metadata['n_classes']
        class_names = metadata['class_names']
        
        print(f'Number of leads: {n_leads}')
        print(f'Number of timesteps: {n_timesteps}')
        print(f'Number of classes: {n_classes}')
        print(f'Classes: {class_names}')
    except:
        print('Metadata not found. Please run preprocessing first.')
        return
    
    # Load datasets
    from torch.utils.data import DataLoader
    
    train_dataset = torch.load(os.path.join(args.output_dir, 'data', 'train_dataset.pt'))
    val_dataset = torch.load(os.path.join(args.output_dir, 'data', 'val_dataset.pt'))
    test_dataset = torch.load(os.path.join(args.output_dir, 'data', 'test_dataset.pt'))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Train GAN
    if args.mode in ['train_gan', 'all']:
        print('Training LSTM-GAN...')
        from src.models.generator import ECGGenerator
        from src.models.discriminator import ECGDiscriminator
        from src.training.train_gan import train_gan
        
        # Initialize models
        generator = ECGGenerator(
            noise_dim=args.noise_dim,
            class_dim=n_classes,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            leads=n_leads,
            ecg_length=n_timesteps
        ).to(device)
        
        discriminator = ECGDiscriminator(
            class_dim=n_classes,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            leads=n_leads
        ).to(device)
        
        # Train GAN
        gan_history = train_gan(
            generator=generator,
            discriminator=discriminator,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            noise_dim=args.noise_dim,
            n_epochs=args.n_epochs,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            save_interval=args.save_interval,
            sample_interval=args.sample_interval,
            log_interval=args.log_interval,
            results_dir=os.path.join(args.output_dir, 'results'),
            models_dir=os.path.join(args.output_dir, 'models')
        )
    
    # Train baseline models
    if args.mode in ['train_baseline', 'all']:
        print('Training baseline models...')
        from src.models.lstm_classifier import LSTMClassifier
        from src.training.train_baseline import train_and_evaluate
        from src.utils.sampling import AugmentedDataLoader
        
        # LSTM Classifier
        lstm_classifier = LSTMClassifier(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            leads=n_leads,
            class_dim=n_classes
        ).to(device)
        
        lstm_acc, lstm_f1, lstm_results = train_and_evaluate(
            model=lstm_classifier,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model_name='LSTM_Classifier',
            n_epochs=args.n_epochs,
            lr=args.lr,
            save_dir=os.path.join(args.output_dir, 'baseline')
        )
        
        # LSTM with data augmentation
        lstm_aug = LSTMClassifier(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            leads=n_leads,
            class_dim=n_classes
        ).to(device)
        
        # Create augmented data loader
        aug_train_loader = AugmentedDataLoader(train_loader)
        
        lstm_aug_acc, lstm_aug_f1, lstm_aug_results = train_and_evaluate(
            model=lstm_aug,
            train_loader=aug_train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model_name='LSTM_Augmented',
            n_epochs=args.n_epochs,
            lr=args.lr,
            save_dir=os.path.join(args.output_dir, 'baseline')
        )
    
    # Evaluate models
    if args.mode in ['evaluate', 'all']:
        print('Evaluating models...')
        from src.evaluation.evaluate import evaluate_discriminator, compare_models
        
        # Load trained models
        from src.models.generator import ECGGenerator
        from src.models.discriminator import ECGDiscriminator
        
        generator = ECGGenerator(
            noise_dim=args.noise_dim,
            class_dim=n_classes,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            leads=n_leads,
            ecg_length=n_timesteps
        ).to(device)
        
        discriminator = ECGDiscriminator(
            class_dim=n_classes,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            leads=n_leads
        ).to(device)
        
        # Load trained weights
        generator.load_state_dict(torch.load(os.path.join(args.output_dir, 'models', 'generator_final.pt')))
        discriminator.load_state_dict(torch.load(os.path.join(args.output_dir, 'models', 'discriminator_final.pt')))
        
        # Evaluate discriminator
        accuracy, f1, all_preds, all_labels, all_probs, all_features = evaluate_discriminator(
            discriminator=discriminator,
            test_loader=test_loader,
            class_names=class_names,
            device=device,
            output_dir=os.path.join(args.output_dir, 'evaluation')
        )
        
        # Generate samples
        from src.utils.visualization import plot_generated_samples
        
        plot_generated_samples(
            generator=generator,
            noise_dim=args.noise_dim,
            n_classes=n_classes,
            class_names=class_names,
            device=device,
            save_dir=os.path.join(args.output_dir, 'evaluation')
        )
        
        # Compare with baseline models
        compare_models(
            gan_results_path=os.path.join(args.output_dir, 'evaluation', 'evaluation_results.pkl'),
            lstm_results_path=os.path.join(args.output_dir, 'baseline', 'LSTM_Classifier_results.pkl'),
            lstm_aug_results_path=os.path.join(args.output_dir, 'baseline', 'LSTM_Augmented_results.pkl'),
            class_names=class_names,
            output_dir=os.path.join(args.output_dir, 'evaluation')
        )
    
    print('Done!')

if __name__ == '__main__':
    main()