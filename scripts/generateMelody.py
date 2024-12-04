import numpy as np
import soundfile as sf

def generate_simple_melody(file_name, frequencies, durations, sr=22050, target_duration=15):
    melody_duration = sum(durations)

    repetitions = int(np.ceil(target_duration / melody_duration))
    extended_frequencies = frequencies * repetitions
    extended_durations = durations * repetitions

    melody = []
    total_generated_duration = 0 

    for freq, duration in zip(extended_frequencies, extended_durations):
        if total_generated_duration + duration > target_duration:
            duration = target_duration - total_generated_duration
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        melody.append(0.5 * np.sin(2 * np.pi * freq * t))
        total_generated_duration += duration

        if total_generated_duration >= target_duration:
            break

    melody = np.hstack(melody)

    sf.write(file_name, melody, sr)

frequencies = [
    261.63, 329.63, 392.00, 523.25, 392.00, 329.63, 261.63, 293.66, 349.23, 
    440.00, 349.23, 293.66, 261.63, 329.63, 392.00, 523.25
]  # Notes from C major scale with variations

durations = [
    0.5, 0.5, 0.5, 0.75, 0.25, 0.5, 0.5, 
    0.75, 0.25, 0.5, 0.5, 1.0
]

generate_simple_melody("simple_melody_15sec.wav", frequencies, durations)
