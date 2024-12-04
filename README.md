# WubAi: Fine-Tuning Transformers for Targeted EDM Track Generation

Electronic Dance Music (EDM) has widespread use and ejoyment but production
often requires significant expertise and time. This project leverages deep learning
to automate the generation of EDM tracks by fine-tuning the pre-trained MusicGen
model. By freezing encoder layers, the model was trained using a composite loss
function combining style and melody metrics, with audio features extracted via
Librosa. The training dataset, derived from the Free Music Archive, was tailored
to house/techno sub-genres, with input melodies generated algorithmically. Quan-
titative evaluation showed somewhat observable reductions in feature deviations
and DTW costs, reflecting small stylistic and melodic improvements across epochs.
Qualitative analysis highlighted increasingly complex and engaging tracks, albeit
with occasional missteps in harmonic balance. This work demonstrates the po-
tential of transformer-based models for targeted music generation and highlights
opportunities for further refinement in audio feature use for training.
