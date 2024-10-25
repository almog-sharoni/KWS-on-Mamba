#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch_optimizer import Lookahead


from data_loader import load_speech_commands_dataset, load_bg_noise_dataset
from utils import set_memory_GB,print_model_size, log_to_file
from augmentations import add_time_shift_and_align, add_silence
from train_utils import trainig_loop





# In[3]:


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


# In[4]:


torch.cuda.is_available()


# In[5]:


# set_memory_GB(1)


# In[6]:


train_ds, val_ds, test_ds, silence_ds , info = load_speech_commands_dataset()
# bg_noise_ds = load_bg_noise_dataset()
bg_noise_ds = None


# In[7]:


# maintain seed for repructablity
np.seed = 42
# tf.random.set_seed(42)
torch.manual_seed(0)


# In[8]:


label_names = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
print(label_names)


# In[9]:


augmentations = [
    lambda x: add_time_shift_and_align(x),
]


# In[10]:


import torch
import numpy as np
import random
from torch.utils.data import Dataset
from librosa.feature import mfcc, delta

class TFDatasetAdapter(Dataset):
    def __init__(self, tf_dataset, bg_noise_dataset, fixed_length, n_mfcc, n_fft, hop_length, n_mels, augmentation=False, derivative=True, noise_level=0.3, MFCC_transform=True, quantize_8bit=False):
        self.tf_dataset = tf_dataset
        self.data = list(tf_dataset)
        self.bg_noise_data = list(bg_noise_dataset) if bg_noise_dataset is not None else None
        self.fixed_length = fixed_length
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.augmentation = augmentation
        self.derivative = derivative
        self.noise_level = noise_level
        self.MFCC_transform = MFCC_transform
        self.quantize_8bit = quantize_8bit  # New parameter for 8-bit quantization

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio, label = self.data[idx]
        audio = audio.numpy()

        # Normalize the audio tensor
        audio = audio / np.max(np.abs(audio))

        # Convert to float
        audio = audio.astype(np.float32)

        # Ensure the audio tensor has the correct shape (1D array)
        if audio.ndim > 1:
            audio = np.squeeze(audio)

        # Add noise from bg_noise data
        if self.bg_noise_data:
            bg_noise_audio = random.choice(self.bg_noise_data)

            # Trim or pad bg_noise to match the audio length
            if len(bg_noise_audio) < len(audio):
                bg_noise_audio = np.pad(bg_noise_audio, (0, len(audio) - len(bg_noise_audio)), mode='constant')
            else:
                # Take a random slice of bg_noise_audio with the same length as the original audio
                start_idx = random.randint(0, len(bg_noise_audio) - len(audio))
                bg_noise_audio = bg_noise_audio[start_idx:start_idx + len(audio)]

            # Add bg_noise as noise to the original audio
            audio = audio + self.noise_level * bg_noise_audio

        # Pad or trim the audio to the fixed length
        if len(audio) < self.fixed_length:
            audio = np.pad(audio, (0, self.fixed_length - len(audio)), mode='constant')
        else:
            audio = audio[:self.fixed_length]

        output = audio

        # Apply augmentations if any
        if self.augmentation:
            for aug in self.augmentation:
                audio = aug(audio)

        # Apply MFCC transformation if enabled
        if self.MFCC_transform:
            audio = audio.astype(np.float32)
            MFCC = mfcc(y=audio, sr=16000, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
            
            if self.derivative:
                # Create MFCC first and second-order deltas
                MFCC_delta = delta(MFCC)
                MFCC_delta2 = delta(MFCC, order=2)

                # Stack MFCC with its deltas
                MFCC = np.vstack([MFCC, MFCC_delta, MFCC_delta2])

            # Remove extra dimension if it exists
            if output.ndim == 3:
                MFCC = MFCC.squeeze(-1)

            output = MFCC

        # Apply 8-bit quantization if the option is enabled
        if self.quantize_8bit:
            output = (output * 127).astype(np.int8)  # Scale float32 to int8 range (-128 to 127)

        return torch.tensor(output, dtype=torch.float32 if not self.quantize_8bit else torch.int8), torch.tensor(label.numpy(), dtype=torch.long)


# In[11]:


# Convert the TFDS dataset to a PyTorch Dataset
fixed_length = 16000
n_mfcc = 13
n_fft = 640
hop_length = 80
n_mels = 100
#take just 10 of the dataset
train_ds = train_ds.take(1000)
val_ds = val_ds.take(100)

# Initialize datasets with configurations
pytorch_train_dataset = TFDatasetAdapter(train_ds, bg_noise_ds, **dataset, augmentation=[lambda x: add_time_shift_and_align(x)])
pytorch_val_dataset = TFDatasetAdapter(val_ds, None, **dataset, augmentation=None)


# In[12]:


# Create a DataLoader to feed the data into the model
batch_size = 32
train_loader = DataLoader(pytorch_train_dataset, batch_size=batch_size, shuffle=True,num_workers=4,prefetch_factor=2)
val_loader = DataLoader(pytorch_val_dataset, batch_size=batch_size, shuffle=False,num_workers=4,prefetch_factor=2)


# # Quantized Mamba

# In[13]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import repeat
import ipdb

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
            in_channels=self.d_model,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_model,  # Use depthwise convolution
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

    def forward(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        # assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        # Input projection
        xz = self.in_proj(hidden_states.squeeze(1))  # (B, 2D)
        x, z = xz.chunk(2, dim=-1)  # Split into x and z

        # Convolution update
        # Print debug information before updating conv_state
        print(f"Shape of x: {x.shape}")
        print(f"Shape of conv_state before update: {conv_state.shape}")
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)  # Rolling state update
        print(f"Shape of conv_state after roll: {conv_state.shape}")
        conv_state[:, :, :] = x
        print(f"Shape of conv_state: {conv_state.shape}")
        print(f"Shape of x: {x.shape}")

        # Use F.conv1d instead of element-wise multiplication
        # conv1d expects input shape (B, C, L), so we reshape conv_state appropriately
        x = self.conv1d(conv_state).to(dtype=dtype)
        x = self.act(x).to(dtype=dtype)

        # Check the shape of x before x_proj
        print(f"Shape of x before x_proj: {x.shape}")

        # Reshape x to ensure it matches the input shape of self.x_proj
        B, C, L = x.shape
        print(f"B: {B}, C: {C}, L: {L}")
        
        # Reshape x and keep track of changes in batch size
        x = x.permute(0, 2, 1).reshape(B * L, C)
        print(f"Shape of x after reshape: {x.shape}")

        # Projection for SSM input
        x_db = self.x_proj(x)
        print(f"Shape of x_db after x_proj: {x_db.shape}")  # Debug the shape after projection

        # Split x_db into dt, B, and C components
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Print the shape of dt and other tensors
        print(f"Shape of dt before addition: {dt.shape}")
        print(f"Shape of self.dt_proj.bias: {self.dt_proj.bias.shape}")

        # Project dt to correct shape
        dt = self.dt_proj(dt)
        dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))

        # SSM step: dt, A, and B are combined to update the state
        A = -torch.exp(self.A_log.float())
        print(f"Shape of A: {A.shape}")

        dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
        print(f"Shape of dA: {dA.shape}")

        dB = torch.einsum("bd,bn->bdn", dt, B)
        print(f"Shape of dB: {dB.shape}")

        # Before performing operations, check the shapes of ssm_state, dA, and dB
        print(f"Shape of ssm_state: {ssm_state.shape}")
        print(f"Shape of dA: {dA.shape}")
        print(f"Shape of dB: {dB.shape}")
        print(f"Shape of x: {x.shape}")

        # Reshape ssm_state if the batch size has changed due to reshaping
        if ssm_state.shape[0] != dA.shape[0]:
            print(f"Reshaping ssm_state from {ssm_state.shape[0]} to {dA.shape[0]}")
            ssm_state = ssm_state.repeat_interleave(L, dim=0)

        # Ensure that x has the same batch size as ssm_state
        if x.shape[0] != ssm_state.shape[0]:
            print(f"Reshaping x from batch size {x.shape[0]} to match ssm_state batch size {ssm_state.shape[0]}")
            x = x.repeat_interleave(L, dim=0)

        # Perform the SSM update step
        ssm_state = ssm_state = ssm_state * dA + x.unsqueeze(2) * dB
        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
        y = y + self.D.to(dtype) * x
        # Reshape y back to 3D tensor before multiplication
        batch_size = z.size(0)  # 32
        seq_len = y.size(0) // batch_size  # Calculate the sequence length
        y = y.view(batch_size, seq_len, -1)  # Reshape y to [32, seq_len, 204]
        # Ensure z is aligned with y before multiplication
        # Align sequence length of `y` to match `z`
        if y.size(1) > z.size(1):
            y = y[:, :z.size(1), :]  # Slice y to match z's sequence length
            print(f"Slicing y to match z: {y.shape}")
        elif y.size(1) < z.size(1):
            # Expand or pad `y` to match `z`, if necessary
            padding = z.size(1) - y.size(1)
            y = F.pad(y, (0, 0, 0, padding))  # Pad y along the sequence length dimension
            print(f"Padding y to match z: {y.shape}")
        print(f"Shape of y: {y.shape}")
        print(f"Shape of z: {z.shape}")
        # Perform element-wise multiplication between y and z
        y = y * self.act(z)  # Ensure batch sizes match here
        
        # Output projection
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state




# In[14]:


# # Initialize the module
# d_model = 64  # Dimensionality of the model input/output
# module = QuantizedMambaModule(d_model=d_model, device='cpu', dtype=torch.float32)

# # Create dummy input hidden states
# # Assuming we want to process a batch of 10 samples with input dimensionality d_model
# hidden_states = torch.randn(10, 1, d_model, device=module.device, dtype=module.dtype)

# # Updated initial convolution state
# # The shape of conv_state's last dimension should match the kernel size of the convolution layer
# conv_state = torch.zeros(10, module.d_inner, module.conv1d.kernel_size[0], device=module.device, dtype=module.dtype)

# # Create initial SSM state
# ssm_state = torch.zeros(10, module.d_inner, module.d_state, device=module.device, dtype=module.dtype)

# # Perform a forward pass
# output, updated_conv_state, updated_ssm_state = module(hidden_states, conv_state, ssm_state)

# # Print the outputs
# print("Output:", output)
# print("Updated Convolution State:", updated_conv_state)
# print("Updated SSM State:", updated_ssm_state)


# # RNN based SSM

# In[15]:


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
            if self.conv_state is None:
                self.conv_state = torch.zeros(x.size(0), x.size(1), mamba_layer.d_inner, device=x.device, dtype=x.dtype)
            if self.ssm_state is None:
                self.ssm_state = torch.zeros(x.size(0), mamba_layer.d_inner, mamba_layer.d_state, device=x.device, dtype=x.dtype)
            x, self.conv_state, self.ssm_state = mamba_layer(x, self.conv_state, self.ssm_state)
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


# In[16]:


# configs = {'d_state': 64, 'd_conv': 10, 'expand': 2, 'batch_size': 26, 'dropout_rate': 0.134439213335519, 'num_mamba_layers': 2, 'n_mfcc': 10, 'n_fft': 200, 'hop_length': 160, 'n_mels': 30, 'noise_level': 0.2582577623788829, 'lr': 0.0011942156978344588, 'weight_decay': 2.5617519345807027e-05}
# dataset = {
#     'fixed_length': 16000,
#     'n_mfcc': configs['n_mfcc'],  # Use from configs
#     'n_fft': configs['n_fft'],    # Use from configs
#     'hop_length': configs['hop_length'],  # Use from configs
#     'n_mels': configs['n_mels'],  # Use from configs
#     'noise_level': configs['noise_level']  # Use from configs
# }
# model_configs = {

#     'input_dim': configs['n_mfcc'] * 3,  # Use from configs
#     'd_model': (dataset['fixed_length'] // dataset['hop_length']) + 1 + 1,  # Use from configs
#     'd_state': configs['d_state'],  # Use from configs
#     'd_conv': configs['d_conv'],    # Use from configs
#     'expand': configs['expand'],    # Use from configs
#     'num_mamba_layers': configs['num_mamba_layers'],  # Use from configs
#     'dropout_rate': configs['dropout_rate'],  # Use from configs
#     'label_names': ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
# }


# # Initialize datasets with configurations
# pytorch_train_dataset = TFDatasetAdapter(train_ds, bg_noise_ds, **dataset, augmentation=[lambda x: add_time_shift_and_align(x)])
# pytorch_val_dataset = TFDatasetAdapter(val_ds, None, **dataset, augmentation=None)
# # Create a DataLoader to feed the data into the model
# batch_size = 32
# train_loader = DataLoader(pytorch_train_dataset, batch_size=batch_size, shuffle=True,num_workers=4,prefetch_factor=2)
# val_loader = DataLoader(pytorch_val_dataset, batch_size=batch_size, shuffle=False,num_workers=4,prefetch_factor=2)


# In[17]:


mamba_model = KeywordSpottingModel_with_cls(**model_configs).to('cuda')



# In[18]:


# for x, y in train_loader:
#     print(x.shape)
#     model.eval()
#     predicted = torch.argmax(model(x), dim=1)
#     print(predicted)
#     break


# In[19]:


states = torch.load('best_model_95.pth')

# new_state_dict = mamba_model.state_dict()

# for state in states:
#     print(state)
    


# In[20]:


# for state in new_state_dict:
#     print(state)


# In[21]:


mamba_model.load_state_dict(states,strict=True)


# In[10]:





# In[22]:


# load test data
pytorch_test_dataset = TFDatasetAdapter(test_ds,None,dataset['fixed_length'],dataset['n_mfcc'],dataset['n_fft'],dataset['hop_length'],dataset['n_mels'],augmentation=None)
test_loader = DataLoader(pytorch_test_dataset, batch_size= configs['batch_size'], shuffle=False,num_workers=9,prefetch_factor=2,drop_last=True)


# In[ ]:





# # Test Eval

# In[23]:


# Evaluate the model on the test set

accuracy = 0
total = 0
mamba_model.eval()

with torch.no_grad():
    for audio, labels in test_loader:
        audio, labels = audio.to("cuda"), labels.to("cuda")
        outputs = mamba_model(audio)
        outputs = torch.abs(outputs)
        aggregated_output = outputs.mean(dim=1)  # shape: [batch_size, num_classes]

        # Step 2: Apply softmax to get probabilities
        probabilities = torch.softmax(aggregated_output, dim=-1)  # shape: [batch_size, num_classes]

        # Step 3: Get the class with the highest probability
        predicted = torch.argmax(probabilities, dim=-1)  # shape: [batch_size]
        print(predicted)
        total += labels.size(0)
        accuracy += (predicted == labels).sum().item()
        break
test_accuracy = 100 * accuracy / total
print(f'Test Accuracy: {test_accuracy}%')


# In[74]:


for name, param in mamba_model.named_parameters():
    if param.requires_grad:
        print(f"Weight of {name}: {param.data}")


# # Compare outputs of Mamba

# In[ ]:


from 


# In[ ]:




