import torch
import librosa
import numpy as np
from audiocraft.models import MusicGen
from audiocraft.data.audio_utils import convert_audio
from scipy.signal import resample

model = MusicGen.get_pretrained('melody')

def evaluate_melody_similarity(output_audio, input_melody):
    y_out, sr_out = librosa.load(output_audio, sr=None)
    y_in, sr_in = librosa.load(input_melody, sr=None)

    output_pitch, _ = librosa.piptrack(y=y_out, sr=sr_out)
    input_pitch, _ = librosa.piptrack(y=y_in, sr=sr_in)
    output_pitch_max = np.nan_to_num(np.max(output_pitch, axis=0))
    input_pitch_max = np.nan_to_num(np.max(input_pitch, axis=0))

    min_length = min(len(output_pitch_max), len(input_pitch_max))
    output_pitch_max = resample(output_pitch_max, min_length)
    input_pitch_max = resample(input_pitch_max, min_length)

    D, wp = librosa.sequence.dtw(output_pitch_max[:, np.newaxis], input_pitch_max[:, np.newaxis])
    return D[-1, -1]

def evaluate_style(output_audio, reference_audio):
    y_out, sr_out = librosa.load(output_audio, sr=None)
    y_ref, sr_ref = librosa.load(reference_audio, sr=None)

    centroid_out = librosa.feature.spectral_centroid(y=y_out, sr=sr_out)
    centroid_ref = librosa.feature.spectral_centroid(y=y_ref, sr=sr_ref)

    min_length = min(centroid_out.shape[1], centroid_ref.shape[1])
    centroid_out_resampled = resample(centroid_out, min_length, axis=1)
    centroid_ref_resampled = resample(centroid_ref, min_length, axis=1)

    centroid_diff = np.mean(np.abs(centroid_out_resampled - centroid_ref_resampled))
    return centroid_diff

def custom_loss(output_audio, input_melody, reference_audio):
    melody_similarity_penalty = evaluate_melody_similarity(output_audio, input_melody)
    style_similarity_penalty = evaluate_style(output_audio, reference_audio)
    return melody_similarity_penalty + 0.5 * style_similarity_penalty

def fine_tune(model, dataloader, optimizer):
    model.train()
    for batch in dataloader:
        input_melody, reference_audio = batch['melody'], batch['reference']
        
        output_audio = model.generate(input_melody)

        loss = custom_loss(output_audio, input_melody, reference_audio)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")

dataloader = torch.utils.data.DataLoader("", batch_size=1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
fine_tune(model, dataloader, optimizer)
