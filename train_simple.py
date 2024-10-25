import torch
from model import KeywordSpottingModel_with_cls

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
model1 = KeywordSpottingModel_with_cls(**model_configs).to('cuda')

input_data = torch.randn(32, 69, 135).to('cuda')

# Forward pass through both models
model1.eval()
_ = model1(input_data)