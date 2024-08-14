import torch.nn as nn
from mamba_ssm import Mamba

# Define model architecture
class KeywordSpottingModel(nn.Module):
    def __init__(self, input_dim, d_model, d_state, d_conv, expand, label_names):
        super(KeywordSpottingModel, self).__init__()
        self.proj = nn.Linear(input_dim, d_model)  # Initial projection layer
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.fc = nn.Linear(d_model, len(label_names))  # Output layer

        self.dropout = nn.Dropout(0.5)  # Dropout layer with a dropout rate of 0.5


    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape to [batch_size, num_frames, num_mfcc]
        x = self.proj(x)  # Project input to d_model dimension
        x = x.permute(0, 2, 1)  # Transpose to [batch_size, d_model, num_frames] for Mamba
        x = self.mamba(x)
        x = self.dropout(x)  # Apply dropout after Mamba
        x = x.mean(dim=2)  # Global average pooling over the time dimension
        x = self.fc(x)
        return x