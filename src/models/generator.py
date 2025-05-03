import torch
import torch.nn as nn

class ECGGenerator(nn.Module):
    def __init__(self, noise_dim, class_dim, hidden_dim=128, num_layers=2, leads=12, ecg_length=1000):
        super(ECGGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.class_dim = class_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.leads = leads
        self.ecg_length = ecg_length
        
        # Initial fully connected layer to process noise and class label
        self.fc_input = nn.Linear(noise_dim + class_dim, hidden_dim)
        
        # LSTM to generate sequence
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Output layer to generate ECG signal
        self.fc_output = nn.Linear(hidden_dim * 2, leads)  # *2 for bidirectional
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        
    def forward(self, z, labels):
        batch_size = z.size(0)
        
        # Concatenate noise and labels
        x = torch.cat([z, labels], dim=1)
        
        # Process through initial FC layer
        x = self.fc_input(x)
        x = self.leaky_relu(x)
        
        # Reshape for LSTM input: (batch_size, seq_len, input_size)
        # Create a sequence of the same vector repeated for each time step
        x = x.unsqueeze(1).repeat(1, self.ecg_length, 1)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Generate ECG signal
        ecg = self.fc_output(lstm_out)
        ecg = self.tanh(ecg)
        
        # Reshape to (batch_size, leads, timesteps)
        ecg = ecg.permute(0, 2, 1)
        
        return ecg