import torch.nn as nn
import sys
## import Mamba from mamba/mamba_ssm/modules/mamba_simple.py
sys.path.append('mamba/mamba_ssm/modules')
# from mamba_simple import MambaQuantized as Mamba
from mamba_simple import Mamba
import torch

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
    
    # Define model architecture
import torch
import torch.nn as nn
import torch.quantization

class KeywordSpottingModel_with_cls(nn.Module):
    def __init__(self, input_dim, d_model, d_state, d_conv, expand, label_names, num_mamba_layers=1, dropout_rate=0.2):
        super(KeywordSpottingModel_with_cls, self).__init__()
        
        # Initial projection layer
        self.proj = nn.Linear(input_dim, d_model)  
        
        # CLS token: learnable parameter with shape [1, 1, d_model]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Quantization stubs
        self.quant = torch.quantization.QuantStub()  # Quantize the input
        self.dequant = torch.quantization.DeQuantStub()  # Dequantize output if needed
        
        # Stack multiple Mamba layers with RMSNorm layer
        self.mamba_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_mamba_layers):
            self.mamba_layers.append(Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand))
            self.layer_norms.append(nn.modules.normalization.RMSNorm(d_model))

        # Output layer
        self.fc = nn.Linear(d_model, len(label_names))  
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Quantize the input
        # x = self.quant(x)

        # Reshape to [batch_size, num_frames, num_mfcc]
        x = x.permute(0, 2, 1)
        
        # # Dequantize before projection to ensure dtype match with weights
        # x = self.dequant(x)
        
        # Project input to d_model dimension
        x = self.proj(x)  
        
        # # Re-quantize after projection (optional, based on your quantization strategy)
        # x = self.quant(x)
        
        # Create a CLS token and expand it across the batch dimension
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: [batch_size, 1, d_model]
        
        # Append the CLS token to the input sequence
        x = torch.cat((x, cls_tokens), dim=1)  # Shape: [batch_size, num_frames + 1, d_model]
        x = x.permute(0, 2, 1)  # Transpose to [batch_size, d_model, num_frames + 1] for Mamba
        
        # Pass through Mamba layers and layer normalization
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            x = mamba_layer(x)
            x = layer_norm(x)  # Apply RMSNorm after Mamba layer

        x = self.dropout(x)  # Apply dropout after Mamba layers
        
        # Extract the CLS token output (last token)
        cls_output = x[:, :, -1]  # Shape: [batch_size, d_model]
        
        # Dequantize before the final fully connected layer
        # cls_output = self.dequant(cls_output)
        
        # Pass through the output layer
        x = self.fc(cls_output)
        
        return x