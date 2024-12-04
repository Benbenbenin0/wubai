import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from utils import evaluate_generated_track

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("techno_tuned_musicgen_hf_4", map_location=device)
model.eval()
melody_path = "data/simple_melody_30sec.wav"
melody, sr = librosa.load(melody_path, sr=44100)
melody_tensor = torch.tensor(melody).unsqueeze(0).to(device)

print("Generating track...")
with torch.no_grad():
    generated_audio = model.generate(inputs={"melody": melody_tensor, "melody_sampling_rate": sr}, max_new_tokens=1024)
    generated_audio = generated_audio.squeeze().cpu().numpy()

librosa.output.write_wav("generated_track.wav", generated_audio, sr)
print("Generated track saved to 'generated_track.wav'.")

target_edm_path = "data/sample_target_edm.wav"
target_audio, target_sr = librosa.load(target_edm_path, sr=44100)

target_features = {
    "spectral_centroid_mean": 5000,
    "tempo": 128
}

evaluation_results = evaluate_generated_track(generated_audio, target_audio, target_features)

print("Evaluation Metrics:")
for key, value in evaluation_results.items():
    print(f"{key}: {value:.4f}")

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 2)
plt.title("Spectral Centroid Over Time")
gen_centroid = librosa.feature.spectral_centroid(y=generated_audio, sr=sr)[0]
target_centroid = librosa.feature.spectral_centroid(y=target_audio, sr=target_sr)[0]
plt.plot(gen_centroid, label="Generated Track", alpha=0.7)
plt.plot(target_centroid, label="Target Track", alpha=0.7)
plt.legend()
plt.xlabel("Frames")
plt.ylabel("Centroid (Hz)")

plt.subplot(3, 1, 3)
plt.title("DTW Alignment Cost Matrix")
import librosa.display
D, wp = librosa.sequence.dtw(X=librosa.feature.mfcc(y=generated_audio, sr=sr).T,
                             Y=librosa.feature.mfcc(y=target_audio, sr=target_sr).T, metric="euclidean")
librosa.display.specshow(D, x_axis="frames", y_axis="frames", cmap="viridis")
plt.plot(wp[:, 1], wp[:, 0], label="Warping Path", color="red")
plt.legend()
plt.xlabel("Generated Track Frames")
plt.ylabel("Target Track Frames")

plt.tight_layout()
plt.savefig("evaluation_results.png")
plt.show()

print("Evaluation graphs saved as 'evaluation_results.png'.")
