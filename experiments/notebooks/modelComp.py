import torch
from src.models.model import KeywordSpottingModel_with_cls

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None


class QuantizedMambaModule(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device='None',
        dtype=None,
    ):
        super(QuantizedMambaModule, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # Store device and dtype as class attributes
        self.device = device
        self.dtype = dtype
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # Depthwise Convolution (quantizable)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,  # Use depthwise convolution
            bias=conv_bias,
            **factory_kwargs
        )

        # Activation
        self.act = nn.SiLU()

        # Projection for SSM inputs
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)

        # Projection for dt (time-step related parameters)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize dt projection weight
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias (between dt_min and dt_max)
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # **Updated** S4D state initialization (A matrix for SSM)
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),  # Create on the correct device
            "n -> d n",
            d=self.d_inner
        ).contiguous()

        # Take the log of A, keeping it as a floating-point tensor (fp32)
        A_log = torch.log(A)  # Shape will be [d_inner, d_state]
        self.A_log = nn.Parameter(A_log)  # Register as a parameter
        self.A_log._no_weight_decay = True  # Set weight decay attribute


        # Skip connection D
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # D parameter for skip connection
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)


    def forward(self, hidden_states):
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        """
        Simplified forward function for standard inference without `inference_params`.
        Args:
            hidden_states: (B, L, D) Input tensor for inference.
        Returns:
            Output tensor after passing through the model.
        """
        batch, seqlen, dim = hidden_states.shape

        # Directly proceed with your model's forward operations
        print(f"type(self.in_proj): {type(self.in_proj)}")
        print(f"type(self.in_proj.weight): {type(self.in_proj.weight)}")

        xz = rearrange(self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"), "d (b l) -> b d l", l=seqlen)
        
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        x, z = xz.chunk(2, dim=1)

        x = self.act(self.conv1d(x)[..., :seqlen])
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        A = -torch.exp(self.A_log.float())
        y = selective_scan_fn(x, dt, A, B, C, self.D.float(), z=z,
                            delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)

        return out


import torch.quantization


class KeywordSpottingModel_with_cls_custom(nn.Module):
    def __init__(self, input_dim, d_model, d_state, d_conv, expand, label_names, num_mamba_layers=1, dropout_rate=0.2):
        super(KeywordSpottingModel_with_cls_custom, self).__init__()
        
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
            self.mamba_layers.append(QuantizedMambaModule(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,device='cuda', dtype=torch.float32))
            self.layer_norms.append(nn.modules.normalization.RMSNorm(d_model))

        # Output layer
        self.fc = nn.Linear(d_model, len(label_names))  
        self.dropout = nn.Dropout(dropout_rate)

        self.conv_state = None

        self.ssm_state = None

    def forward(self, x):
        # Quantize the input
        # x = self.quant(x)

        # Reshape to [batch_size, num_frames, num_mfcc]
        x = x.permute(0, 2, 1)
        
        # # Dequantize before projection to ensure dtype maQch with weights
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
            # if self.conv_state is None:
            #     self.conv_state = torch.zeros(x.size(0), x.size(1), mamba_layer.d_inner, device=x.device, dtype=x.dtype)
            # if self.ssm_state is None:
            #     self.ssm_state = torch.zeros(x.size(0), mamba_layer.d_inner, mamba_layer.d_state, device=x.device, dtype=x.dtype)
            x = mamba_layer(x)
            x = layer_norm(x)  # Apply RMSNorm after Mamba layer

        x = self.dropout(x)  # Apply dropout after Mamba layers
        
        # Extract the CLS token output (last token)
        cls_output = x[:, :, -1]  # Shape: [batch_size, d_model]
        
        # # Dequantize before the final fully connected layer
        # cls_output = self.dequant(cls_output)
        
        # Pass through the output layer
        x = self.fc(cls_output)
        print(f"Shape of x: {x.shape}")
        return x


configs = {'d_state': 51, 'd_conv': 10, 'expand': 2, 'batch_size': 26, 'dropout_rate': 0.134439213335519, 'num_mamba_layers': 2, 'n_mfcc': 23, 'n_fft': 475, 'hop_length': 119, 'n_mels': 61, 'noise_level': 0.2582577623788829, 'lr': 0.0011942156978344588, 'weight_decay': 2.5617519345807027e-05}


dataset = {
    'fixed_length': 16000,
    'n_mfcc': configs['n_mfcc'],  # Use from configs
    'n_fft': configs['n_fft'],    # Use from configs
    'hop_length': configs['hop_length'],  # Use from configs
    'n_mels': configs['n_mels'],  # Use from configs
    'noise_level': configs['noise_level']  # Use from configs
}


model_configs = {

    'input_dim': configs['n_mfcc'] * 3,  # Use from configs
    'd_model': (dataset['fixed_length'] // dataset['hop_length']) + 1 + 1,  # Use from configs
    'd_state': configs['d_state'],  # Use from configs
    'd_conv': configs['d_conv'],    # Use from configs
    'expand': configs['expand'],    # Use from configs
    'num_mamba_layers': configs['num_mamba_layers'],  # Use from configs
    'dropout_rate': configs['dropout_rate'],  # Use from configs
    'label_names': ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
}
# init models
model1 = KeywordSpottingModel_with_cls_custom(**model_configs).to('cuda')
model2 = KeywordSpottingModel_with_cls(**model_configs).to('cuda')



# Define a dictionary to store outputs for each model
model1_outputs = {}
model2_outputs = {}

# Define hook function to capture outputs
def hook_fn(name, outputs_dict):
    def hook(module, input, output):
        outputs_dict[name] = output.detach().cuda().numpy() if isinstance(output, torch.Tensor) else output
    return hook

# Attach hooks to each layer of both models
def attach_hooks(model, outputs_dict):
    for name, layer in model.named_modules():
        layer.register_forward_hook(hook_fn(name, outputs_dict))

# Assuming `model1` and `model2` are your two Mamba models
attach_hooks(model1, model1_outputs)
attach_hooks(model2, model2_outputs)

# # Create some random input data (make sure it's consistent for both models)
# # input_data = torch.randn(1,136,69)

# input_data = torch.randn(32, 69, 135).to('cuda')

# # maintain reproducibility
# torch.manual_seed(0)

# # Forward pass through both models
# _ = model1(input_data)
# _ = model2(input_data)


# # Now model1_outputs and model2_outputs contain the outputs of each layer

# # Compare outputs layer by layer
# for layer_name in model1_outputs:
#     if layer_name in model2_outputs:
#         output1 = model1_outputs[layer_name]
#         output2 = model2_outputs[layer_name]
        
#         # Compute difference (example: Mean Squared Error)
#         mse = ((output1 - output2) ** 2).mean()
#         print(f"Layer {layer_name} MSE: {mse}")
#     else:
#         print(f"Layer {layer_name} not found in second model.")

#  import test
from src.data.data_loader import load_speech_commands_dataset, TFDatasetAdapter
from torch.utils.data import DataLoader
from config import dataset, data_loader, model as model_config, optimizer as optimizer_config, scheduler as scheduler_config, training

train_ds, val_ds, test_ds, train_silence, info = load_speech_commands_dataset(version=3, reduced=True)

# test_ds = test_ds.take(100)

#load model
device = 'cuda'

model = KeywordSpottingModel_with_cls_custom(**model_config).to(device)
model.load_state_dict(torch.load("best_model_95.pth"))
# load test data
pytorch_test_dataset = TFDatasetAdapter(test_ds, None, **dataset, augmentation=None)

# # Apply dynamic quantization to the model
# # This will quantize only nn.Linear and nn.LSTM layers by default
# quantized_model = torch.quantization.quantize_dynamic(
#     model,  # Model to quantize
#     {nn.Linear},  # Specify the layers to quantize (e.g., nn.Linear)
#     dtype=torch.qint8  # Use 8-bit integer weights
# )

# # Set the model to evaluation mode
# quantized_model.eval()
# model = quantized_model
test_loader =  DataLoader(pytorch_test_dataset, **data_loader, shuffle=False)


# Evaluate the model on the test set
accuracy = 0
total = 0
# Set the model to evaluation mode
model.eval()


with torch.no_grad():
    for audio, labels in test_loader:
        audio, labels = audio.to(device), labels.to(device)
        outputs = model(audio)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        accuracy += (predicted == labels).sum().item()
test_accuracy = 100 * accuracy / total
print(f'Test Accuracy: {test_accuracy}%')



