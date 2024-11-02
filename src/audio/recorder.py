import pyaudio
import wave  # Add this import
import time
import whisper

# Previous setup code stays the same
audio = pyaudio.PyAudio()
format = pyaudio.paInt16
channels = 1
rate = 44100
chunk = 1024
frames = []

stream = audio.open(
    format=format,
    channels=channels,
    rate=rate,
    input=True,
    frames_per_buffer=chunk
)

# Recording code
chunks_per_second = int(rate / chunk)
recording_seconds = 5
total_chunks = chunks_per_second * recording_seconds

print(f"Recording for {recording_seconds} seconds...")

for i in range(total_chunks):
    data = stream.read(chunk)
    frames.append(data)
    if i % chunks_per_second == 0:
        print(f"Second {i//chunks_per_second + 1}...")

print("Done recording")

# Clean up the audio stream
stream.stop_stream()
stream.close()
audio.terminate()

# Now let's save the file
# Can you guess what 'wb' means? (hint: write binary)
with wave.open('my_recording.wav', 'wb') as wave_file:
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(audio.get_sample_size(format))
    wave_file.setframerate(rate)
    wave_file.writeframes(b''.join(frames))

print("Saved as my_recording.wav")

def transcribe_audio(file_path):
    model = whisper.load_model("large-v3-turbo")
    result = model.transcribe(file_path)
    return result["text"]

audio_file = "my_recording.wav"
print("Transcribing audio...")
transcription = transcribe_audio(audio_file)
print("Transcriptiion", transcription)
