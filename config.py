# config.py

dataset = {
    'fixed_length': 16000,
    'n_mfcc': 13,
    'n_fft': 640,
    'hop_length': 320,
    'n_mels': 40,
    'noise_level': 0.05
}

data_loader = {
    'batch_size': 32,
    'num_workers': 4,
    'prefetch_factor': 2
}

model = {
    'input_dim': 39,
    'd_model': 52,
    'd_state': 16,
    'd_conv': 4,
    'expand': 2,
    'num_mamba_layers': 1,
    'dropout_rate': 0.1,
    'label_names': ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
}

optimizer = {
    'lr': 0.0024,
    'weight_decay': 2.80475e-05,
    'lookahead': {
        'k': 5,
        'alpha': 0.5
    }
}

scheduler = {
    'reduce_lr_on_plateau': {
        'mode': 'min',
        'factor': 0.1,
        'patience': 3
    }
}

training = {
    'num_epochs': 100
}