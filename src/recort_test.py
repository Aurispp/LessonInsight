import pyaudio
import wave
import time
import threading
import sys
from datetime import datetime

class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000

    def start_recording(self):
        self.is_recording = True
        self.frames = []
        
        # Open audio stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        print("Recording... Type 'stop' and press Enter to finish")
        
        # Record until stopped
        while self.is_recording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)

    def stop_recording(self):
        self.is_recording = False
        
        # Stop and close the stream
        self.stream.stop_stream()
        self.stream.close()
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"../audio_files/recording_{timestamp}.wav"
        
        # Save the recorded audio
        print(f"\nSaving to {filename}")
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        return filename

    def __del__(self):
        self.audio.terminate()

def main():
    recorder = AudioRecorder()
    
    # Start recording in a separate thread
    record_thread = threading.Thread(target=recorder.start_recording)
    record_thread.start()
    
    # Wait for 'stop' command
    while True:
        if input().lower() == 'stop':
            filename = recorder.stop_recording()
            print(f"Recording saved to: {filename}")
            break
    
    record_thread.join()

if __name__ == "__main__":
    main()