import os
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class EDMDataset(Dataset):
    def __init__(self, edm_dir, features_csv, melody_path, sr=44100):
        self.edm_files = sorted([os.path.join(edm_dir, f) for f in os.listdir(edm_dir) if f.endswith(".wav")])
        self.features = pd.read_csv(features_csv)
        self.melody, melody_sr = librosa.load(melody_path, sr=None)
        if melody_sr != sr:
            self.melody = librosa.resample(self.melody, orig_sr=melody_sr, target_sr=sr)

        self.sr = sr
        self.target_length = 30 * sr

        print(f"Number of audio files: {len(self.edm_files)}")
        print(f"Number of feature rows: {len(self.features)}")
        assert len(self.edm_files) == len(self.features), "Mismatch between audio files and feature rows."

    def __len__(self):
        return len(self.edm_files)

    def __getitem__(self, idx):
        edm_track, _ = librosa.load(self.edm_files[idx], sr=self.sr)

        if len(edm_track) < self.target_length:
            edm_track = np.pad(edm_track, (0, self.target_length - len(edm_track)), 'constant')
        else:
            edm_track = edm_track[:self.target_length]

        target_features = self.features.iloc[idx].apply(
            lambda x: float(x.strip("[]")) if isinstance(x, str) and x.startswith("[") else float(x)
        ).to_numpy(dtype=np.float32)

        return (
            torch.tensor(self.melody, dtype=torch.float32),
            torch.tensor(edm_track, dtype=torch.float32),
            torch.tensor(target_features, dtype=torch.float32)
        )

def get_dataloader(edm_dir, features_csv, melody_path, batch_size=10, shuffle=True):
    dataset = EDMDataset(edm_dir, features_csv, melody_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
