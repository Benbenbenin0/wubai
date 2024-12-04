import os
import librosa
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class AudioDataset:
    def __init__(self, edm_dir, features_csv, melody_path, sr=44100):
        self.edm_files = sorted([os.path.join(edm_dir, f) for f in os.listdir(edm_dir) if f.endswith(".wav")])
        self.features = pd.read_csv(features_csv)
        self.melody, melody_sr = librosa.load(melody_path, sr=None)
        self.sr = sr
        self.target_length = 30 * sr

        if melody_sr != sr:
            self.melody = librosa.resample(self.melody, orig_sr=melody_sr, target_sr=sr)

    def __len__(self):
        return len(self.edm_files)

    def __getitem__(self, idx):
        edm_track, _ = librosa.load(self.edm_files[idx], sr=self.sr)

        if len(edm_track) < self.target_length:
            edm_track = librosa.util.fix_length(edm_track, self.target_length)
        else:
            edm_track = edm_track[:self.target_length]

        target_features = self.features.iloc[idx].to_dict()

        return (
            torch.tensor(self.melody, dtype=torch.float32),
            torch.tensor(edm_track, dtype=torch.float32),
            target_features
        )

    def get_dataloader(self, batch_size=10, shuffle=True):
        dataset = Dataset(self)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
