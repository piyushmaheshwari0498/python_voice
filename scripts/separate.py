import os
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import torch
import soundfile as sf
import numpy as np

# ----------------------------
# 1. Load diarization pipeline
# ----------------------------
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# ----------------------------
# 2. Speaker embedding model
# ----------------------------
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb", device="cpu"
)

# ----------------------------
# 3. Function: get embedding
# ----------------------------
def get_embedding(wav_file):
    waveform, sample_rate = sf.read(wav_file)
    if waveform.ndim > 1:  # stereo → mono
        waveform = np.mean(waveform, axis=1)
    waveform = torch.tensor(waveform).unsqueeze(0)
    return embedding_model({"waveform": waveform, "sample_rate": sample_rate})


# ----------------------------
# 4. Process conversation audio
# ----------------------------
conversation_file = "scripts/conversation_fixed.wav"
mom_reference_file = "scripts/mom.wav"

# diarize the conversation
diarization = pipeline(conversation_file)

# Load audio
audio = AudioSegment.from_wav(conversation_file)

# Get mom's reference embedding
mom_embedding = get_embedding(mom_reference_file)

# ----------------------------
# 5. Extract only mom’s segments
# ----------------------------
threshold = 0.7  # similarity threshold (tune this if needed)
mom_audio = AudioSegment.empty()

for turn, _, speaker in diarization.itertracks(yield_label=True):
    start = int(turn.start * 1000)
    end = int(turn.end * 1000)
    segment = audio[start:end]

    # Save temporary file for embedding
    segment_file = "temp_segment.wav"
    segment.export(segment_file, format="wav")

    seg_embedding = get_embedding(segment_file)

    # Cosine similarity between mom and this segment
    similarity = torch.nn.functional.cosine_similarity(
        mom_embedding, seg_embedding
    ).item()

    if similarity >= threshold:
        print(f"✅ Keeping segment {start/1000:.2f}-{end/1000:.2f}s (similarity={similarity:.2f})")
        mom_audio += segment
    else:
        print(f"❌ Skipping segment {start/1000:.2f}-{end/1000:.2f}s (similarity={similarity:.2f})")

# ----------------------------
# 6. Save final mom-only audio
# ----------------------------
output_file = "scripts/mom_only.wav"
mom_audio.export(output_file, format="wav")

print(f"\n🎉 Saved mom's clean audio as {output_file}")