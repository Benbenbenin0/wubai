import librosa

file_path = "data/simple_melody_30sec.wav"
audio, sr = librosa.load(file_path, sr=None)
print(f"Sample Rate (Hz): {sr}")
