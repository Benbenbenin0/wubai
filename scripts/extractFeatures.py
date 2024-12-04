import os
import librosa
import pandas as pd

generated_dir = "data/generatedTracks"

target_features = {
    "spectral_centroid_mean": 5000,
    "spectral_bandwidth_mean": 2000,
    "tempo": 128,
    "spectral_contrast_mean": 20,
}

def extract_features(audio_path, sr=44100):
    y, _ = librosa.load(audio_path, sr=sr)
    features = {
        "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr).mean(),
        "tempo": librosa.beat.tempo(y=y, sr=sr)[0],
        "spectral_contrast": librosa.feature.spectral_contrast(y=y, sr=sr).mean(),
    }
    return features

results = []
for file_name in sorted(os.listdir(generated_dir)):
    if file_name.endswith(".wav"):
        file_path = os.path.join(generated_dir, file_name)
        epoch = int(file_name.split(".")[0])
        features = extract_features(file_path)
        features["epoch"] = epoch
        features["track"] = file_name
        results.append(features)

df_features = pd.DataFrame(results)
df_features.to_csv("generated_features.csv", index=False)
print("Features extracted and saved to 'generated_features.csv'")
