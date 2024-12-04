import os
import torch
import torchaudio
from utils import load_musicgen_model, get_dataloader, compute_total_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, processor = load_musicgen_model(freeze_layers=True, device=device)

dataloader = get_dataloader(
    edm_dir="data/techno_edm_tracks",
    features_csv="data/processed_techno_features.csv",
    melody_path="data/simple_melody_30sec.wav"
)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

os.makedirs("generatedTracks", exist_ok=True)

# Training loop
print("Starting training:")
for epoch in range(10):  # Number of epochs
    epoch_loss = 0.0
    for batch_idx, (melody, edm_track, edm_features) in enumerate(dataloader):
        edm_track = edm_track.to(device)

        melody_waveform, melody_sample_rate = torchaudio.load("data/simple_melody_30sec.wav")
        if melody_sample_rate != 44100:
            resampler = torchaudio.transforms.Resample(orig_freq=melody_sample_rate, new_freq=44100)
            melody_waveform = resampler(melody_waveform)

        text_prompt = "high-energy techno house EDM track with driving beats"
        inputs = processor(text=text_prompt, padding=True, return_tensors="pt")
        inputs["audio"] = melody_waveform
        inputs["sampling_rate"] = 44100

        optimizer.zero_grad()

        generated_audio = model.generate(**inputs, max_new_tokens=1024)
        output_path = os.path.join("generatedTracks", f"generated_epoch{epoch}_batch{batch_idx}.wav")
        generated_audio_waveform = generated_audio.squeeze(0).cpu()
        torchaudio.save(output_path, generated_audio_waveform.unsqueeze(0), 44100)

        loss = compute_total_loss(generated_audio, edm_track, melody, edm_features)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch} complete. Average Loss: {avg_epoch_loss:.4f}")

model.save_pretrained("techno_tuned_musicgen_hf_4")
processor.save_pretrained("techno_tuned_musicgen_hf_4")
print("Model saved to 'techno_tuned_musicgen_hf_4'.")
