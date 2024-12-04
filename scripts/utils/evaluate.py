import librosa
import numpy as np

def evaluate_generated_track(generated, target, target_features, sr=44100):
    gen_centroid = librosa.feature.spectral_centroid(y=generated, sr=sr).mean()
    target_centroid = librosa.feature.spectral_centroid(y=target, sr=sr).mean()
    centroid_diff = abs(gen_centroid - target_centroid)

    gen_tempo = librosa.beat.tempo(y=generated, sr=sr)[0]
    target_tempo = librosa.beat.tempo(y=target, sr=sr)[0]
    tempo_diff = abs(gen_tempo - target_tempo)

    gen_mfcc = librosa.feature.mfcc(y=generated, sr=sr)
    target_mfcc = librosa.feature.mfcc(y=target, sr=sr)
    _, wp_cost = librosa.sequence.dtw(X=gen_mfcc.T, Y=target_mfcc.T, metric="euclidean")
    dtw_cost = wp_cost.mean()

    return {
        "centroid_diff": centroid_diff,
        "tempo_diff": tempo_diff,
        "dtw_cost": dtw_cost
    }
