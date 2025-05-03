# Generative Adversarial LSTM for End-to-End ECG Diagnosis

This repository contains the implementation of a novel deep learning framework that combines generative adversarial networks (GANs) and LSTM networks for ECG-based diagnosis of cardiac events. Our approach uses a conditional GAN (cGAN) to generate realistic ECG waveforms corresponding to specific cardiac conditions, which helps augment scarce pathological data.

## Authors
- OMRANI Rabah
- Maroc Abdelhakim Fouad
- ESI SBA School

## Overview

Our LSTM-GAN model consists of two main components:
1. **Generator (G)**: A conditional LSTM-based network that produces synthetic ECG signals based on a noise vector and class label.
2. **Discriminator (D)**: A network that both distinguishes real vs. fake signals and predicts the cardiac condition.

The model is trained and evaluated on the PTB-XL dataset, a large publicly available electrocardiography dataset.

![Generated ECG Samples](working/results/generated_samples.png)

## Key Features

- End-to-end deep learning framework for ECG diagnosis
- Conditional GAN with LSTM networks for temporal modeling
- Auxiliary classifier in the discriminator for joint adversarial and classification tasks
- Effective handling of class imbalance through synthetic data generation
- Comprehensive evaluation and comparison with baseline models

## Results

Our LSTM-GAN model outperforms traditional LSTM classifiers and LSTM with data augmentation:

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| LSTM | 0.8911 | 0.8821 |
| LSTM+AugGAN | 0.9234 | 0.9214 |
| LSTM-GAN (Ours) | 0.9143 | 0.9090 |

Improvement over LSTM baseline: Accuracy +2.32%, F1 +2.69%

![Model Comparison](working/evaluation/model_comparison.png)

## Repository Structure

\`\`\`
├── data/                  # Data storage directory
├── src/                   # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # Model architectures
│   ├── training/          # Training procedures
│   ├── evaluation/        # Evaluation scripts
│   └── utils/             # Utility functions
├── results/               # Training results and visualizations
├── models/                # Saved model weights
├── evaluation/            # Evaluation results
├── baseline/              # Baseline model results
├── main.py                # Main script
└── README.md              # This file
\`\`\`

## Installation

```bash
# Clone the repository
git clone https://github.com/username/ecg-lstm-gan.git
cd ecg-lstm-gan

# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

## Usage

### Data Preprocessing

```shellscript
python main.py --mode preprocess --data_path /path/to/ptb-xl-dataset/ --output_dir output
```

### Training

```shellscript
# Train LSTM-GAN
python main.py --mode train_gan --output_dir output --n_epochs 15

# Train baseline models
python main.py --mode train_baseline --output_dir output --n_epochs 15
```

### Evaluation

```shellscript
python main.py --mode evaluate --output_dir output
```

### Run all steps

```shellscript
python main.py --mode all --data_path /path/to/ptb-xl-dataset/ --output_dir output --n_epochs 15
```

## Citation

If you use this code in your research, please cite our work:

```plaintext
@article{omrani2023generative,
  title={Generative Adversarial LSTM for End-to-End ECG Diagnosis},
  author={Omrani, Rabah and Fouad, Maroc Abdelhakim},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
