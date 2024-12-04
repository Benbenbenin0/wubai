import os
import pandas as pd
import librosa
from tqdm import tqdm

def extract_tempos(wav_dir, csv_file, output_csv):
    if not os.path.isdir(wav_dir):
        print(f"Error: Directory '{wav_dir}' does not exist.")
        return

    if not os.path.isfile(csv_file):
        print(f"Error: CSV file '{csv_file}' does not exist.")
        return

    df = pd.read_csv(csv_file)
    tempos = []

    print("Extracting tempos...")
    for track_id in tqdm(df['track_id'], desc="Processing tracks"):
        file_name = f"{int(track_id):06d}.wav"
        wav_file = os.path.join(wav_dir, file_name)

        if not os.path.isfile(wav_file):
            print(f"Warning: File '{wav_file}' not found. Skipping...")
            tempos.append(None)
            continue

        try:
            y, sr = librosa.load(wav_file, sr=None)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempos.append(tempo)
        except Exception as e:
            print(f"Error processing '{wav_file}': {e}")
            tempos.append(None)

    df['tempo'] = tempos
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved to '{output_csv}'.")

def print_tempos(csv_file):
    try:
        df = pd.read_csv(csv_file)
        if 'tempo' not in df.columns:
            print("Error: 'tempo' column not found in the CSV file.")
            return

        print("Track tempos:")
        for index, row in df.iterrows():
            track_id = row['track_id']
            tempo = row['tempo']
            print(f"Track ID: {track_id}, Tempo: {tempo} BPM")

    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    wav_directory = "data/techno_edm_tracks"
    input_csv = "data/techno_features_flat.csv"
    output_csv = "data/techno_features_flat_with_tempo.csv"
    extract_tempos(wav_directory, input_csv, output_csv)
    print_tempos(output_csv)
