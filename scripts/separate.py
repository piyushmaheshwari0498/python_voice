import sys
import os
import soundfile as sf
from pyannote.audio import Pipeline
from pydub import AudioSegment

# --- Helper: compute embedding for a reference audio file ---
def get_embedding(wav_file):
    if not os.path.exists(wav_file):
        raise FileNotFoundError(
            f"❌ Missing reference file: {wav_file}. Please commit it or upload it as an artifact."
        )
    waveform, sample_rate = sf.read(wav_file)
    return waveform, sample_rate

# --- MAIN SCRIPT ---
if len(sys.argv) < 2:
    print("Usage: python scripts/separate.py <input_audio.wav>")
    sys.exit(1)

audio_file = sys.argv[1]
mom_reference_file = "scripts/mom.wav"   # reference voice file

# Load HF token
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=os.environ.get("HF_TOKEN")
)

print("Running speaker diarization, please wait...")
diarization = pipeline(audio_file)
print("✅ Diarization completed!")

# Load original audio
audio = AudioSegment.from_wav(audio_file)

# Collect speaker segments
speaker_segments = {}
for turn, _, speaker in diarization.itertracks(yield_label=True):
    segment = audio[int(turn.start * 1000): int(turn.end * 1000)]
    if speaker not in speaker_segments:
        speaker_segments[speaker] = segment
    else:
        speaker_segments[speaker] += segment

# Save segments
output_dir = "scripts"
os.makedirs(output_dir, exist_ok=True)

for speaker, segment in speaker_segments.items():
    filename = os.path.join(output_dir, f"{speaker}.wav")
    segment.export(filename, format="wav")
    print(f"💾 Saved {filename}")

# Save mom's voice separately (longest duration heuristic)
mom_speaker = max(speaker_segments, key=lambda s: len(speaker_segments[s]))
mom_only_file = os.path.join(output_dir, "mom_only.wav")
speaker_segments[mom_speaker].export(mom_only_file, format="wav")
print(f"🎙️ Saved mom's voice as {mom_only_file}")

# Ensure mom.wav exists for embeddings
if os.path.exists(mom_reference_file):
    waveform, sr = get_embedding(mom_reference_file)
    print(f"✅ Verified mom reference file {mom_reference_file} (sr={sr}, shape={waveform.shape})")
else:
    print(f"⚠️ No mom reference file found at {mom_reference_file}. Skipping embedding check.")

print("✨ All done!")