import pandas as pd
import os
import librosa
import soundfile as sf
from tqdm import tqdm

genres = pd.read_csv('data/genres.csv')
tracks = pd.read_csv('data/tracks.csv', low_memory=False)

# Filter for EDM genres
# edm_genres = ['Techno']  # Add sub-genres as needed
# edm_ids = genres[genres['title'].isin(edm_genres)]['genre_id']

target_genre_ids = [181, 182, 184, 296, 337]
target_genre_top = "Electronic"
edm_tracks = tracks[
    (tracks['genres'].apply(lambda x: any(genre_id in eval(x) for genre_id in target_genre_ids) if isinstance(x, str) else False)) &
    (tracks['genre_top'] == target_genre_top)
]

output_dir = 'techno_edm_tracks'
os.makedirs(output_dir, exist_ok=True)

print("Processing audio files...")
for _, row in tqdm(edm_tracks.iterrows(), total=len(edm_tracks), desc="Converting MP3 to WAV"):
    track_id = str(int(row['track_id'])).zfill(6)
    parent_dir = track_id[:3]
    track_path = f"data/fma_large/{parent_dir}/{track_id}.mp3"
    
    if os.path.exists(track_path):
        output_path = os.path.join(output_dir, f"{track_id}.wav")
        try:
            audio, sr = librosa.load(track_path, sr=None)
            sf.write(output_path, audio, sr)
        except Exception as e:
            print(f"Error processing {track_path}: {e}")
    else:
        print(f"File not found: {track_path}")

features = pd.read_csv('data/features.csv', skiprows=3, low_memory=False)

print("Saving EDM track features...")
edm_features = features[features['track_id'].isin(edm_tracks['track_id'])]
edm_features.to_csv('techno_features.csv', index=False)

print("Processing complete.")
