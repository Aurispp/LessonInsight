import pyaudio
import wave
import threading
from datetime import datetime
from pathlib import Path

current_file = Path(__file__)  
print(f"Current file location: {current_file}")

project_root = current_file.parent.parent.parent
print(f"Project root: {project_root}")

# Set audio directory path
audio_dir = project_root / "audio_files"
print(f"Audio directory: {audio_dir}")
# Setup audio
audio = pyaudio.PyAudio()
format = pyaudio.paInt16
channels = 1
rate = 44100
chunk = 1024
frames = []
is_recording = True

def stop_recording():
    global is_recording
    input("Press Enter to stop recording...\n")
    is_recording = False

print("Starting setup...")

try:
    stream = audio.open(
        format=format,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk
    )
    print("Audio stream opened successfully")
except Exception as e:
    print(f"Error opening stream: {e}")
    exit()

# Start a thread to watch for the stop signal
stop_thread = threading.Thread(target=stop_recording)
stop_thread.start()

print("Recording... Press Enter to stop")

# Record until Enter is pressed
while is_recording:
    try:
        data = stream.read(chunk)
        frames.append(data)
    except Exception as e:
        print(f"Error during recording: {e}")
        break

print("Recording finished! Saving file...")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = audio_dir / f"recording_{timestamp}.wav"
with wave.open(str(filename), 'wb') as wave_file:
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(audio.get_sample_size(format))
    wave_file.setframerate(rate)
    wave_file.writeframes(b''.join(frames))
    
print(f"Saved as {filename}")
