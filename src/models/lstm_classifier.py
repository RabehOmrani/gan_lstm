import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=2, leads=12, class_dim=5):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.leads = leads
        self.class_dim = class_dim
        
        # LSTM to process ECG signal
        self.lstm = nn.LSTM(
            input_size=leads,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim * 2, class_dim)  # *2 for bidirectional
        
    def forward(self, ecg):
        # Reshape input: (batch_size, leads, timesteps) -> (batch_size, timesteps, leads)
        x = ecg.permute(0, 2, 1)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        features = lstm_out[:, -1, :]
        
        # Class prediction
        class_logits = self.fc(features)
        
        return class_logits, features