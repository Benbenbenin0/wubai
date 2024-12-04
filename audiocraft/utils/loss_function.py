import torch
import librosa

def compute_loss(generated, edm_track, edm_features, melody):
    style_loss = compute_style_loss(generated, edm_features)
    melody_loss = compute_melody_loss(generated, melody)
    return 0.5 * style_loss + 0.5 * melody_loss

def compute_style_loss(generated, edm_features):
    gen_audio = generated.squeeze().cpu().numpy()
    gen_centroid = librosa.feature.spectral_centroid(y=gen_audio).mean()
    gen_bandwidth = librosa.feature.spectral_bandwidth(y=gen_audio).mean()
    gen_tempo = librosa.beat.tempo(y=gen_audio)[0]

    centroid_loss = (gen_centroid - edm_features["spectral_centroid_mean"]) ** 2
    bandwidth_loss = (gen_bandwidth - edm_features["spectral_bandwidth_mean"]) ** 2
    tempo_loss = (gen_tempo - edm_features["tempo"]) ** 2

    return centroid_loss + bandwidth_loss + tempo_loss

def compute_melody_loss(generated, melody):
    gen_audio = generated.squeeze().cpu().numpy()
    melody_audio = melody.cpu().numpy()

    gen_mfcc = librosa.feature.mfcc(y=gen_audio)
    melody_mfcc = librosa.feature.mfcc(y=melody_audio)

    _, dtw_cost = librosa.sequence.dtw(X=gen_mfcc.T, Y=melody_mfcc.T)
    return dtw_cost.mean()
