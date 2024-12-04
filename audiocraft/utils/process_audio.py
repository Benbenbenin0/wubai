import torch
import torchaudio
import librosa
import numpy as np

def preprocess_audio(audio_tensor, output_path, sample_rate=44100):
    if audio_tensor.device != torch.device("cpu"):
        audio_tensor = audio_tensor.cpu()
    audio_tensor = audio_tensor.squeeze(0)

    audio_numpy = audio_tensor.numpy()

    max_val = np.max(np.abs(audio_numpy))
    if max_val > 0:
        audio_numpy = audio_numpy / max_val

    torchaudio.save(output_path, torch.tensor(audio_numpy).unsqueeze(0), sample_rate)

    print(f"Audio saved to {output_path}")
