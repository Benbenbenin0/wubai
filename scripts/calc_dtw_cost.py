import os
import librosa
import numpy as np
import pandas as pd

melody_path = "data/simple_melody_15sec.wav"
generated_dir = "data/generatedTracks"

melody, sr = librosa.load(melody_path, sr=44100)
melody_mfcc = librosa.feature.mfcc(y=melody, sr=sr)

def compute_dtw_cost(generated_path, melody_mfcc):
    generated_audio, _ = librosa.load(generated_path, sr=44100)
    generated_mfcc = librosa.feature.mfcc(y=generated_audio, sr=44100)
    _, dtw_cost = librosa.sequence.dtw(X=generated_mfcc.T, Y=melody_mfcc.T, metric="euclidean")
    return dtw_cost.mean()

results = []
for epoch in range(1, 11):
    track_path = os.path.join(generated_dir, f"{epoch}.wav")
    if os.path.exists(track_path):
        dtw_cost = compute_dtw_cost(track_path, melody_mfcc)
        results.append({"epoch": epoch, "dtw_cost": dtw_cost})
    else:
        print(f"File {epoch}.wav not found in {generated_dir}")

dtw_df = pd.DataFrame(results)
dtw_df.to_csv("dtw_costs.csv", index=False)
print("DTW costs calculated and saved to 'dtw_costs.csv'.")
