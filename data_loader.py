import tensorflow_datasets as tfds
from torch.utils.data import DataLoader, Dataset
from librosa.feature import mfcc, delta
import numpy as np
import torch
import augmentations
import random
import os
import librosa

def load_bg_noise_dataset():
    # Path to the UrbanSound8K dataset
    dataset_path = 'UrbanSound8K/UrbanSound8K/audio'

    bgNoise = []

    # Loop through all folds and load audio files
    for fold in range(1, 10):  # Folds are from fold1 to fold9
        fold_path = os.path.join(dataset_path, f'fold{fold}')
        
        # List all audio files in the current fold
        for file_name in os.listdir(fold_path):
            if file_name.endswith('.wav'):  # Process only .wav files
                file_path = os.path.join(fold_path, file_name)
                
                # Load the audio file
                audio, sample_rate = librosa.load(file_path, sr=None)
                # normalize audio
                audio = audio / np.max(np.abs(audio))
                bgNoise.append(audio)
    return bgNoise

def load_speech_commands_dataset(version=3):
    """Load the Speech Commands dataset using TensorFlow Datasets."""
    ds, info = tfds.load(f'speech_commands:0.0.{version}', with_info=True, as_supervised=True, shuffle_files=False)
    train_ds = ds['train']
    val_ds = ds['validation']
    test_ds = ds['test']

    train_silence = train_ds.filter(lambda x, y: y == 10)

    # Filter out the 'unknown' and 'silence' labels
    # If label = 10,11 drop
    train_ds = train_ds.filter(lambda x, y: y != 10 and y != 11)
    val_ds = val_ds.filter(lambda x, y: y != 10 and y != 11)
    test_ds = test_ds.filter(lambda x, y: y != 10 and y != 11)
    return train_ds, val_ds, test_ds, train_silence, info

# Define the dataset adapter:
# Define the dataset adapter:
class TFDatasetAdapter(Dataset):
    def __init__(self, tf_dataset, bg_noise_dataset, fixed_length, n_mfcc, n_fft, hop_length, n_mels, augmentation=False, derivative=True, noise_level=0.3, MFCC_transform=True):
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
            # bg_noise_audio = bg_noise_audio.numpy().astype(np.float32)

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


        if self.MFCC_transform:
        # Create MFCCs from an audio tensor using Librosa.
            audio = audio.astype(np.float32)
            MFCC = mfcc(y=audio, sr=16000, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
            if self.derivative:
                # Create MFCC second, first order delta
                MFCC_delta = delta(MFCC)
                MFCC_delta2 = delta(MFCC, order=2)

                # Stack the three MFCCs together
                MFCC = np.vstack([MFCC, MFCC_delta, MFCC_delta2])

            # Remove extra dimension if it exists
            if output.ndim == 3:
                MFCC = MFCC.squeeze(-1)

            output = MFCC

        return torch.tensor(output, dtype=torch.float32), torch.tensor(label.numpy(), dtype=torch.long)    




# class TFDatasetAdapter(Dataset):
#     def __init__(self, tf_dataset, silence_dataset, fixed_length, n_mfcc, n_fft, hop_length, n_mels, augmentation=False, derivative=True, noise_level=0.3):
#         self.tf_dataset = tf_dataset
#         self.data = list(tf_dataset)
#         self.silence_data = list(silence_dataset) if silence_dataset is not None else None
#         self.fixed_length = fixed_length
#         self.n_mfcc = n_mfcc
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.n_mels = n_mels
#         self.augmentation = augmentation
#         self.derivative = derivative
#         self.noise_level = noise_level

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         audio, label = self.data[idx]
#         audio = audio.numpy()

#         # Convert to float
#         audio = audio.astype(np.float32)

#         # Ensure the audio tensor has the correct shape (1D array)
#         if audio.ndim > 1:
#             audio = np.squeeze(audio)
        
#         # Add noise from silence data
#         if self.silence_data:
#             silence_audio, _ = random.choice(self.silence_data)
#             silence_audio = silence_audio.numpy().astype(np.float32)

#             # Trim or pad silence to match the audio length
#             if len(silence_audio) < len(audio):
#                 silence_audio = np.pad(silence_audio, (0, len(audio) - len(silence_audio)), mode='constant')
#             else:
#                 silence_audio = silence_audio[:len(audio)]

#             # Add silence as noise to the original audio
#             audio = audio + self.noise_level * silence_audio

#         # Apply augmentations if any
#         if self.augmentation:
#             for aug in self.augmentation:
#                 audio = aug(audio)

#         # Pad or trim the audio to the fixed length
#         if len(audio) < self.fixed_length:
#             audio = np.pad(audio, (0, self.fixed_length - len(audio)), mode='constant')
#         else:
#             audio = audio[:self.fixed_length]

#         # Create MFCCs from an audio tensor using Librosa.
#         audio = audio.astype(np.float32)
#         MFCC = mfcc(y=audio, sr=16000, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
#         if self.derivative:
#             # Create MFCC second, first order delta
#             MFCC_delta = delta(MFCC)
#             MFCC_delta2 = delta(MFCC, order=2)

#             # Stack the three MFCCs together
#             MFCC = np.vstack([MFCC, MFCC_delta, MFCC_delta2])

#         # Remove extra dimension if it exists
#         if MFCC.ndim == 3:
#             MFCC = MFCC.squeeze(-1)

#         return torch.tensor(MFCC, dtype=torch.float32), torch.tensor(label.numpy(), dtype=torch.long)    



