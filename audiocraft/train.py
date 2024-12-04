import os
import torch
from audiocraft.models import MusicGen
from audiocraft.data import AudioDataset
from utils import compute_loss, preprocess_audio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MusicGen.get_pretrained('melody')
model.to(device)

dataloader = AudioDataset(
    edm_dir="data/techno_edm_tracks",
    features_csv="data/processed_techno_features.csv",
    melody_path="data/simple_melody_30sec.wav"
).get_dataloader(batch_size=10, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

os.makedirs("generatedTracks", exist_ok=True)

print("Starting training:")
for epoch in range(10):
    epoch_loss = 0.0
    for batch_idx, (melody, edm_track, edm_features) in enumerate(dataloader):
        melody = melody.to(device)
        edm_track = edm_track.to(device)

        optimizer.zero_grad()

        generated_audio = model.generate(melody, description="high-energy techno house EDM track with driving beats")

        output_path = os.path.join("generatedTracks", f"generated_epoch{epoch}_batch{batch_idx}.wav")
        preprocess_audio(generated_audio, output_path)

        loss = compute_loss(generated_audio, edm_track, edm_features, melody)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch} complete. Average Loss: {avg_epoch_loss:.4f}")

model.save("techno_tuned_model_1.pth")
print("Model saved to 'techno_tuned_model_1.pth'.")
