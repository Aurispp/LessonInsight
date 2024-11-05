from pathlib import Path

current_file = Path(__file__)
audio_files_dir = current_file.parent.parent.parent / "audio_files"

wav_files = list(audio_files_dir.glob("*.wav"))

print("\nAvailable recordings: ")
for index, file in enumerate(wav_files):
    print(f"- [{index}] {file.name}")



