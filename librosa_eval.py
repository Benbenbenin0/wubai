import librosa
import numpy as np
from scipy.signal import resample
from librosa.sequence import dtwmelod

def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    pitch, mag = librosa.piptrack(y=y, sr=sr)
    pitch_max = np.max(pitch, axis=0)
    pitch_max[pitch_max == 0] = np.nan
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return {"tempo": tempo, "pitch": pitch_max, "centroid": centroid}

def evaluate_melody_similarity(output_features, reference_features):
    output_pitch = np.nan_to_num(output_features["pitch"])
    reference_pitch = np.nan_to_num(reference_features["pitch"])

    min_length = min(len(output_pitch), len(reference_pitch))
    output_pitch = librosa.util.fix_length(output_pitch, size=min_length)
    reference_pitch = librosa.util.fix_length(reference_pitch, size=min_length)

    D, wp = dtw(output_pitch[:, np.newaxis], reference_pitch[:, np.newaxis], metric='euclidean')
    distance = D[-1, -1]  # Total alignment cost
    return distance

def evaluate_style(output_features, reference_features):
    min_length = min(output_features["centroid"].shape[1], reference_features["centroid"].shape[1])
    output_centroid = resample(output_features["centroid"], min_length, axis=1)
    reference_centroid = resample(reference_features["centroid"], min_length, axis=1)

    centroid_diff = np.mean(np.abs(output_centroid - reference_centroid))
    tempo_diff = abs(output_features["tempo"] - reference_features["tempo"])
    return {"centroid_diff": centroid_diff, "tempo_diff": tempo_diff}

def main():
    output_path = "./data/country.mp3"
    reference_path = "./data/edm_noremix.mp3"

    output_features = extract_features(output_path)
    reference_features = extract_features(reference_path)

    melody_similarity = evaluate_melody_similarity(output_features, reference_features)
    style_similarity = evaluate_style(output_features, reference_features)

    print("\n--- Evaluation Results ---")
    print(f"Melody Similarity (DTW Distance): {melody_similarity:.2f}")
    print(f"Style Similarity:")
    print(f"  - Centroid Difference: {style_similarity['centroid_diff']:.2f}")
    print(f"  - Tempo Difference: {style_similarity['tempo_diff'][0]:.2f}")

if __name__ == "__main__":
    main()
