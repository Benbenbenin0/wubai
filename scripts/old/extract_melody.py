import librosa
import numpy as np
from scipy.signal import medfilt
from scipy.io.wavfile import write

def extract_melody(input_audio_path, output_audio_path, tuning=True):
    y, sr = librosa.load(input_audio_path, sr=None)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = np.max(pitches, axis=0)
    time_frames = np.arange(len(pitch_values))
    threshold = np.percentile(magnitudes, 75) 
    pitch_values[magnitudes.max(axis=0) < threshold] = 0

    pitch_values_smoothed = medfilt(pitch_values, kernel_size=5)

    if tuning:
        pitch_values_smoothed[pitch_values_smoothed > 0] = librosa.hz_to_midi(pitch_values_smoothed[pitch_values_smoothed > 0])
        pitch_values_smoothed[pitch_values_smoothed > 0] = librosa.midi_to_hz(np.round(pitch_values_smoothed[pitch_values_smoothed > 0]))

    melody_waveform = librosa.tone(pitch_values_smoothed, sr=sr, duration=len(y) / sr)

    melody_waveform = melody_waveform / np.max(np.abs(melody_waveform))

    write(output_audio_path, sr, (melody_waveform * 32767).astype(np.int16))
    print(f"Extracted melody saved to: {output_audio_path}")

input_audio = ""
output_audio = ""
extract_melody(input_audio, output_audio, tuning=True)

