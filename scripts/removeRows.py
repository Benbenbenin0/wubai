import os
import pandas as pd

audio_dir = 'data/techno_edm_tracks'
features_csv = 'data/techno_features_flat_with_tempo.csv'
output_csv = 'data/processed_techno_features.csv'

audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
audio_ids = {os.path.splitext(f)[0].lstrip('0') for f in audio_files}  # Remove leading zeros

print(f"Number of audio files: {len(audio_ids)}")

features = pd.read_csv(features_csv)
filtered_features = features[features['track_id'].astype(str).isin(audio_ids)]

print(f"Number of matching feature rows: {len(filtered_features)}")

filtered_features.to_csv(output_csv, index=False)
print(f"Filtered features saved to: {output_csv}")
