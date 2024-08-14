import tensorflow_datasets as tfds
from torch.utils.data import DataLoader, Dataset
from librosa.feature import mfcc
import numpy as np
import torch

def load_speech_commands_dataset(version=3):
    """Load the Speech Commands dataset using TensorFlow Datasets."""
    ds, info = tfds.load(f'speech_commands:0.0.{version}', with_info=True, as_supervised=True)
    train_ds = ds['train']
    val_ds = ds['validation']
    test_ds = ds['test']
    return train_ds, val_ds, test_ds, info

# Define the dataset adapter:
class TFDatasetAdapter(Dataset):
    def __init__(self, tf_dataset, fixed_length, n_mfcc, n_fft, hop_length, n_mels, augmentations=None):
        self.tf_dataset = tf_dataset
        self.data = list(tf_dataset)
        self.fixed_length = fixed_length
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.augmentations = augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio, label = self.data[idx]
        audio = audio.numpy()

        # Ensure the audio tensor has the correct shape (1D array)
        if audio.ndim > 1:
            audio = np.squeeze(audio)
            
        # Apply augmentations if any
        if self.augmentations:
            for aug in self.augmentations:
                audio = aug(audio)

        # Pad or trim the audio to the fixed length
        if len(audio) < self.fixed_length:
            audio = np.pad(audio, (0, self.fixed_length - len(audio)), mode='constant')
        else:
            audio = audio[:self.fixed_length]

        # Create MFCCs from an audio tensor using Librosa.
        audio = audio.astype(np.float32)
        MFCC = mfcc(y=audio, sr=16000, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)

        # Remove extra dimension if it exists
        if MFCC.ndim == 3:
            MFCC = MFCC.squeeze(-1)

        return torch.tensor(MFCC, dtype=torch.float32), torch.tensor(label.numpy(), dtype=torch.long)
