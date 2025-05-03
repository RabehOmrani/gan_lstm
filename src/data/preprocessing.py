import numpy as np
import pandas as pd
import wfdb
import ast
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import os
import pickle

def load_ptbxl_data(path, sampling_rate=100):
    """
    Load and preprocess the PTB-XL dataset
    
    Args:
        path: Path to the PTB-XL dataset
        sampling_rate: Sampling rate (100 or 500 Hz)
        
    Returns:
        X_train, X_val, X_test: ECG signals
        y_train_onehot, y_val_onehot, y_test_onehot: One-hot encoded labels
        metadata: Dictionary with dataset metadata
    """
    # Load and convert annotation data
    print('Loading annotation data...')
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
    
    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
    
    # Filter records with a single diagnostic superclass
    print('Filtering records with a single diagnostic superclass...')
    Y_filtered = Y[Y.diagnostic_superclass.apply(len) == 1].copy()
    Y_filtered['diagnostic_superclass'] = Y_filtered.diagnostic_superclass.apply(lambda x: x[0])
    
    print('\nDiagnostic superclasses distribution:')
    print(Y_filtered.diagnostic_superclass.value_counts())
    
    # Load the filtered data
    print('\nLoading filtered ECG data...')
    X_filtered = load_raw_data(Y_filtered, sampling_rate, path)
    print(f'Loaded {len(X_filtered)} ECG records')
    
    # Split the data into train, validation, and test sets
    print('\nSplitting data into train, validation, and test sets...')
    # Use stratified fold from the dataset for test set
    test_fold = 10
    X_train_val = X_filtered[np.where(Y_filtered.strat_fold != test_fold)]
    y_train_val = Y_filtered[Y_filtered.strat_fold != test_fold].diagnostic_superclass.values
    X_test = X_filtered[np.where(Y_filtered.strat_fold == test_fold)]
    y_test = Y_filtered[Y_filtered.strat_fold == test_fold].diagnostic_superclass.values
    
    # Further split train_val into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    print(f'Train set: {len(X_train)} records')
    print(f'Validation set: {len(X_val)} records')
    print(f'Test set: {len(X_test)} records')
    
    # Standardize the data
    print('\nStandardizing the data...')
    # Reshape to (n_samples, n_features)
    n_samples_train, n_leads, n_timesteps = X_train.shape
    X_train_reshaped = X_train.reshape(n_samples_train, -1)
    
    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    
    # Apply to validation and test data
    X_val_scaled = scaler.transform(X_val.reshape(len(X_val), -1))
    X_test_scaled = scaler.transform(X_test.reshape(len(X_test), -1))
    
    # Reshape back to original dimensions
    X_train_scaled = X_train_scaled.reshape(n_samples_train, n_leads, n_timesteps)
    X_val_scaled = X_val_scaled.reshape(len(X_val), n_leads, n_timesteps)
    X_test_scaled = X_test_scaled.reshape(len(X_test), n_leads, n_timesteps)
    
    # Encode the labels
    print('\nEncoding the labels...')
    encoder = OneHotEncoder(sparse=False)
    
    # Reshape to 2D array for OneHotEncoder
    y_train_2d = y_train.reshape(-1, 1)
    y_val_2d = y_val.reshape(-1, 1)
    y_test_2d = y_test.reshape(-1, 1)
    
    # Fit and transform
    y_train_onehot = encoder.fit_transform(y_train_2d)
    y_val_onehot = encoder.transform(y_val_2d)
    y_test_onehot = encoder.transform(y_test_2d)
    
    # Get class names and indices
    class_names = encoder.categories_[0]
    class_indices = {name: i for i, name in enumerate(class_names)}
    print(f'Classes: {class_names}')
    
    # Create metadata
    metadata = {
        'class_names': class_names,
        'class_indices': class_indices,
        'n_leads': n_leads,
        'n_timesteps': n_timesteps,
        'n_classes': len(class_names),
        'scaler': scaler,
        'encoder': encoder
    }
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_onehot, y_val_onehot, y_test_onehot, metadata

def load_raw_data(df, sampling_rate, path):
    """
    Load raw ECG data from files
    
    Args:
        df: DataFrame with file paths
        sampling_rate: Sampling rate (100 or 500 Hz)
        path: Base path to the dataset
        
    Returns:
        data: Array of ECG signals
    """
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def apply_oversampling(X, y):
    """
    Apply oversampling to balance the dataset
    
    Args:
        X: ECG signals
        y: Labels
        
    Returns:
        X_balanced: Balanced ECG signals
        y_balanced: Balanced labels
    """
    print('\nApplying oversampling...')
    unique_classes, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)
    
    X_balanced = []
    y_balanced = []
    
    for cls_idx, cls in enumerate(unique_classes):
        cls_indices = np.where(y == cls_idx)[0]
        n_repeat = max_count // len(cls_indices)
        remainder = max_count % len(cls_indices)
        
        sampled_indices = np.tile(cls_indices, n_repeat)
        remainder_indices = np.random.choice(cls_indices, remainder, replace=False)
        all_indices = np.concatenate([sampled_indices, remainder_indices])
        
        X_balanced.append(X[all_indices])
        y_balanced.extend([cls_idx] * len(all_indices))
    
    X_balanced = np.concatenate(X_balanced)
    y_balanced = np.array(y_balanced)
    
    # Convert class indices back to one-hot
    y_balanced_onehot = np.zeros((len(y_balanced), len(unique_classes)))
    for i, cls_idx in enumerate(y_balanced):
        y_balanced_onehot[i, cls_idx] = 1
    
    print('\nBalanced class distribution:')
    print(pd.Series(y_balanced).value_counts())
    
    return X_balanced, y_balanced_onehot

def save_datasets(X_train, X_val, X_test, y_train, y_val, y_test, metadata, output_dir='data'):
    """
    Save preprocessed datasets and metadata
    
    Args:
        X_train, X_val, X_test: ECG signals
        y_train, y_val, y_test: One-hot encoded labels
        metadata: Dictionary with dataset metadata
        output_dir: Output directory
    """
    from src.data.dataset import ECGDataset
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create datasets
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)
    
    # Save datasets
    torch.save(train_dataset, os.path.join(output_dir, 'train_dataset.pt'))
    torch.save(val_dataset, os.path.join(output_dir, 'val_dataset.pt'))
    torch.save(test_dataset, os.path.join(output_dir, 'test_dataset.pt'))
    
    # Save metadata
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f'\nSaved datasets and metadata to {output_dir}')