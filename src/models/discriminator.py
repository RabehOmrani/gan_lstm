import torch
import torch.nn as nn

class ECGDiscriminator(nn.Module):
    def __init__(self, class_dim, hidden_dim=128, num_layers=2, leads=12):
        super(ECGDiscriminator, self).__init__()
        self.class_dim = class_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.leads = leads
        
        # LSTM to process ECG signal
        self.lstm = nn.LSTM(
            input_size=leads,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Fully connected layers for real/fake and classification
        self.fc_validity = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        self.fc_class = nn.Linear(hidden_dim * 2, class_dim)
        
        # Activation function
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, ecg):
        # Reshape input: (batch_size, leads, timesteps) -> (batch_size, timesteps, leads)
        x = ecg.permute(0, 2, 1)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        features = lstm_out[:, -1, :]
        
        # Real/Fake prediction
        validity = self.sigmoid(self.fc_validity(features)).view(-1)
        
        # Class prediction
        class_logits = self.fc_class(features)
        
        return validity, class_logits, features