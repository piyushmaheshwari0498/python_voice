import sys
import os
from pyannote.audio import Pipeline
from pydub import AudioSegment

# Get audio file from argument
if len(sys.argv) < 2:
    print("Please provide an input audio file as argument")
    sys.exit(1)

audio_file = sys.argv[1]

# Load HF token from environment variable
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=os.environ.get("HF_TOKEN")
)

print("Running speaker diarization, please wait...")
diarization = pipeline(audio_file)
print("Diarization completed!")

audio = AudioSegment.from_wav(audio_file)

# Separate speakers
speaker_segments = {}
for turn, _, speaker in diarization.itertracks(yield_label=True):
    segment = audio[int(turn.start * 1000): int(turn.end * 1000)]
    if speaker not in speaker_segments:
        speaker_segments[speaker] = segment
    else:
        speaker_segments[speaker] += segment

# Create output folder
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Identify "mom" as the speaker with the longest total duration
mom_speaker = max(speaker_segments, key=lambda s: len(speaker_segments[s]))
mom_filename = os.path.join(output_dir, "mom.wav")
speaker_segments[mom_speaker].export(mom_filename, format="wav")
print(f"Saved mom's voice as {mom_filename}")

# Export other speakers
for speaker, segment in speaker_segments.items():
    if speaker != mom_speaker:
        filename = os.path.join(output_dir, f"{speaker}.wav")
        segment.export(filename, format="wav")
        print(f"Saved {filename}")

print("All done!")