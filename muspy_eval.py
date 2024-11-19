import muspy

# Load input melody and generated track
input_melody = muspy.read_midi("input.mid")
generated_track = muspy.read_midi("output.mid")

def calculate_note_overlap(input_melody, generated_track):
    input_notes = {note.pitch for track in input_melody.tracks for note in track.notes}
    generated_notes = {note.pitch for track in generated_track.tracks for note in track.notes}
    overlap = input_notes & generated_notes
    return len(overlap) / len(input_notes) if input_notes else 0

def calculate_tempo_consistency(generated_track, edm_reference_tempo=128):
    tempos = [track.tempo for track in generated_track.tracks if track.tempo]
    return all(edm_reference_tempo - 5 <= t <= edm_reference_tempo + 5 for t in tempos)

def calculate_rhythmic_density(track):
    total_duration = sum(note.end - note.start for note in track.notes)
    return total_duration / track.total_time if track.total_time else 0

weights = {
    "melody": 0.4,
    "tempo": 0.2,
    "rhythm": 0.3,
    "instrumentation": 0.1,
}

note_overlap_score = calculate_note_overlap(input_melody, generated_track)
print(f"Note Overlap Score: {note_overlap_score}")

tempo_is_consistent = calculate_tempo_consistency(generated_track)
print(f"Tempo Consistent with EDM: {tempo_is_consistent}")

rhythmic_density = [calculate_rhythmic_density(track) for track in generated_track.tracks]
print(f"Rhythmic Density: {rhythmic_density}")

overall_score = (
    note_overlap_score * weights["melody"] +
    tempo_is_consistent * weights["tempo"] +
    sum(rhythmic_density) / len(rhythmic_density) * weights["rhythm"]
)
print(f"Overall Evaluation Score: {overall_score}")

muspy.plot_pianoroll(generated_track.tracks[0], show=True, title="Generated Track")
muspy.plot_pianoroll(input_melody.tracks[0], show=True, title="Input Melody")

