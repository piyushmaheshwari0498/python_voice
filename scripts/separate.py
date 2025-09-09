import os
import sys
from pyannote.audio import Pipeline
from pydub import AudioSegment

# Check for audio file argument
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

# Load the audio file
audio = AudioSegment.from_wav(audio_file)

# Separate speakers into individual segments
speaker_segments = {}

for turn, _, speaker in diarization.itertracks(yield_label=True):
    segment = audio[int(turn.start * 1000): int(turn.end * 1000)]
    if speaker not in speaker_segments:
        speaker_segments[speaker] = segment
    else:
        speaker_segments[speaker] += segment

# Identify "mom" as the speaker with the longest total duration
mom_speaker = max(speaker_segments, key=lambda s: len(speaker_segments[s]))

# Export mom's audio
mom_filename = "mom.wav"
speaker_segments[mom_speaker].export(mom_filename, format="wav")
print(f"Saved mom's voice as {mom_filename}")

# Optionally export the other speaker(s)
for speaker, segment in speaker_segments.items():
    if speaker != mom_speaker:
        filename = f"{speaker}.wav"
        segment.export(filename, format="wav")
        print(f"Saved {filename}")

print("All done!")