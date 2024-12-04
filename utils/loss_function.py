import torch.nn.functional as F
import librosa
from librosa.sequence import dtw
import torch
import numpy as np

def compute_style_loss(generated, target_features, sr=44100):
    gen_audio = generated.squeeze().cpu().numpy()

    gen_centroid = librosa.feature.spectral_centroid(y=gen_audio, sr=sr).mean()
    centroid_loss = ((gen_centroid - target_features['spectral_centroid_mean']) / target_features['spectral_centroid_std']) ** 2

    gen_bandwidth = librosa.feature.spectral_bandwidth(y=gen_audio, sr=sr).mean()
    bandwidth_loss = ((gen_bandwidth - target_features['spectral_bandwidth_mean']) / target_features['spectral_bandwidth_std']) ** 2

    gen_tempo = librosa.beat.tempo(y=gen_audio, sr=sr)[0]
    tempo_loss = ((gen_tempo - target_features['tempo']) / target_features['tempo_std']) ** 2

    gen_contrast = librosa.feature.spectral_contrast(y=gen_audio, sr=sr).mean()
    contrast_loss = ((gen_contrast - target_features['spectral_contrast_mean']) / target_features['spectral_contrast_std']) ** 2

    style_loss = 0.25 * centroid_loss + 0.25 * bandwidth_loss + 0.25 * tempo_loss + 0.25 * contrast_loss
    return style_loss

def compute_melody_loss(generated, melody, sr=44100):
    gen_audio = generated.cpu().numpy()
    melody_audio = melody.cpu().numpy()

    gen_mfcc = librosa.feature.mfcc(y=gen_audio, sr=sr)
    melody_mfcc = librosa.feature.mfcc(y=melody_audio, sr=sr)
    _, dtw_cost = dtw(gen_mfcc.T, melody_mfcc.T)

    gen_fft = torch.fft.rfft(generated)
    melody_fft = torch.fft.rfft(melody)
    cross_corr = torch.sum(gen_fft * melody_fft.conj()).real
    freq_loss = 1.0 - cross_corr / (torch.norm(gen_fft) * torch.norm(melody_fft))

    melody_loss = 0.5 * torch.tensor(dtw_cost.mean()) + 0.5 * freq_loss
    return melody_loss

def compute_total_loss(generated, target, melody, target_features, sr=44100):
    style_loss = compute_style_loss(generated, target_features, sr=sr)
    melody_loss = compute_melody_loss(generated, melody, sr=sr)
    total_weight = style_loss + melody_loss
    style_weight = style_loss / total_weight
    melody_weight = melody_loss / total_weight

    total_loss = style_weight * style_loss + melody_weight * melody_loss
    return total_loss
