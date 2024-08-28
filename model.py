import torch.nn as nn
from mamba_ssm import Mamba

# Define model architecture
class KeywordSpottingModel(nn.Module):
    def __init__(self, input_dim, d_model, d_state, d_conv, expand, label_names, num_mamba_layers=1):
        super(KeywordSpottingModel, self).__init__()
        self.proj = nn.Linear(input_dim, d_model)  # Initial projection layer

        # Stack multiple Mamba layers with RSMNorm layer
        self.mamba_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_mamba_layers):
            self.mamba_layers.append(Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand))
            self.layer_norms.append(nn.modules.normalization.RMSNorm(d_model))

        self.fc = nn.Linear(d_model, len(label_names))  # Output layer
        self.dropout = nn.Dropout(0.2)  # Dropout layer with a dropout rate of 0.5

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape to [batch_size, num_frames, num_mfcc]
        x = self.proj(x)  # Project input to d_model dimension
        x = x.permute(0, 2, 1)  # Transpose to [batch_size, d_model, num_frames] for Mamba
        
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            x = mamba_layer(x)
            x = layer_norm(x)  # Apply RSMNorm after Mamba layer
        
        x = self.dropout(x)  # Apply dropout after Mamba layers
        x = x.max(dim=2).values # Global max pooling
        x = self.fc(x)
        return x
