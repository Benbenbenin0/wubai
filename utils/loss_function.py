import torch.nn.functional as F
import librosa
from librosa.sequence import dtw
import torch

def compute_style_loss(generated, target_features, sr=44100):
    gen_audio = generated.squeeze().cpu().numpy()
    gen_centroid = torch.tensor(librosa.feature.spectral_centroid(y=gen_audio, sr=sr).mean(), device=generated.device)
    gen_bandwidth = torch.tensor(librosa.feature.spectral_bandwidth(y=gen_audio, sr=sr).mean(), device=generated.device)
    gen_tempo = torch.tensor(librosa.beat.tempo(y=gen_audio, sr=sr)[0], device=generated.device)
    gen_contrast = torch.tensor(librosa.feature.spectral_contrast(y=gen_audio, sr=sr).mean(), device=generated.device)

    target_centroid = torch.tensor(target_features['spectral_centroid_mean'], device=generated.device)
    target_centroid_std = torch.tensor(target_features['spectral_centroid_std'], device=generated.device)
    target_bandwidth = torch.tensor(target_features['spectral_bandwidth_mean'], device=generated.device)
    target_bandwidth_std = torch.tensor(target_features['spectral_bandwidth_std'], device=generated.device)
    target_tempo = torch.tensor(target_features['tempo'], device=generated.device)
    target_tempo_std = torch.tensor(target_features['tempo_std'], device=generated.device)
    target_contrast = torch.tensor(target_features['spectral_contrast_mean'], device=generated.device)
    target_contrast_std = torch.tensor(target_features['spectral_contrast_std'], device=generated.device)

    centroid_loss = ((gen_centroid - target_centroid) / target_centroid_std) ** 2
    bandwidth_loss = ((gen_bandwidth - target_bandwidth) / target_bandwidth_std) ** 2
    tempo_loss = ((gen_tempo - target_tempo) / target_tempo_std) ** 2
    contrast_loss = ((gen_contrast - target_contrast) / target_contrast_std) ** 2

    style_loss = 0.25 * centroid_loss + 0.25 * bandwidth_loss + 0.25 * tempo_loss + 0.25 * contrast_loss
    return style_loss

def compute_melody_loss(generated, melody, sr=44100):
    gen_audio = generated.cpu().numpy()
    melody_audio = melody.cpu().numpy()

    gen_mfcc = librosa.feature.mfcc(y=gen_audio, sr=sr)
    melody_mfcc = librosa.feature.mfcc(y=melody_audio, sr=sr)
    _, dtw_cost = dtw(gen_mfcc.T, melody_mfcc.T)
    dtw_loss = torch.tensor(dtw_cost.mean(), device=generated.device, requires_grad=True)

    gen_fft = torch.fft.rfft(generated)
    melody_fft = torch.fft.rfft(melody)
    cross_corr = torch.sum(gen_fft * melody_fft.conj()).real
    freq_loss = 1.0 - cross_corr / (torch.norm(gen_fft) * torch.norm(melody_fft))

    melody_loss = 0.5 * dtw_loss + 0.5 * freq_loss
    return melody_loss

def compute_total_loss(generated, target, melody, target_features, sr=44100):
    style_loss = compute_style_loss(generated, target_features, sr=sr)
    melody_loss = compute_melody_loss(generated, melody, sr=sr)

    total_weight = style_loss + melody_loss
    style_weight = style_loss / total_weight
    melody_weight = melody_loss / total_weight

    total_loss = style_weight * style_loss + melody_weight * melody_loss
    return total_loss
