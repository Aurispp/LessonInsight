import pyaudio
import wave
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

class LocalRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 48000
        self.chunk = 1024
        self.frames = []
        self.is_recording = False
        self.stream = None
        
        # Set up audio directory
        self.audio_dir = Path(__file__).parent.parent.parent / "audio_files"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def get_macbook_mic(self) -> Optional[int]:
        """Get the MacBook Pro Microphone device ID."""
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        for i in range(num_devices):
            device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
            if "macbook pro microphone" in device_info.get('name', '').lower():
                return i
                
        # Try alternative names
        for i in range(num_devices):
            device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
            name = device_info.get('name', '').lower()
            if "built-in microphone" in name or "built-in input" in name:
                return i
        
        return None

    def list_devices(self):
        """List all available audio input devices."""
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        print("\nAvailable Audio Devices:")
        print("-" * 50)
        
        for i in range(num_devices):
            device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
            print(f"ID {i}: {device_info.get('name')}")
            print(f"   Inputs: {device_info.get('maxInputChannels')}")
            print("-" * 50)

    def audio_callback(self, in_data, frame_count, time_info, status) -> Tuple[bytes, int]:
        """Audio stream callback."""
        if self.is_recording and in_data:
            try:
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                self.frames.append(audio_data.copy())
            except Exception as e:
                print(f"Error in audio callback: {e}")
        return (in_data, pyaudio.paContinue)

    def start_recording(self) -> bool:
        """Start recording from MacBook microphone."""
        try:
            mic_id = self.get_macbook_mic()
            if mic_id is None:
                print("Error: Could not find MacBook microphone")
                return False

            # Get device info
            device_info = self.audio.get_device_info_by_host_api_device_index(0, mic_id)
            print(f"\nUsing microphone: {device_info['name']}")
            
            # Open stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=mic_id,
                frames_per_buffer=self.chunk,
                stream_callback=self.audio_callback
            )
            
            # Start recording
            self.frames = []
            self.is_recording = True
            self.stream.start_stream()
            
            print("\nRecording... Press Enter to stop")
            input("")  # Wait for Enter key
            
            return True
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            return False

    def stop_recording(self):
        """Stop recording and save the file."""
        if not self.is_recording:
            return

        print("\nStopping recording...")
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        # Save the recording
        if self.frames:
            try:
                # Process audio data
                audio_data = np.concatenate(self.frames)
                
                # Normalize
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data / max_val * 0.9
                
                # Convert to stereo
                stereo_data = np.column_stack((audio_data, audio_data))
                final_data = (stereo_data.flatten() * 32767).astype(np.int16)
                
                # Save to file
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = self.audio_dir / f"recording_{timestamp}.wav"
                
                with wave.open(str(filename), 'wb') as wave_file:
                    wave_file.setnchannels(2)  # Save as stereo
                    wave_file.setsampwidth(2)
                    wave_file.setframerate(self.rate)
                    wave_file.writeframes(final_data.tobytes())
                
                print(f"\nRecording saved as: {filename}")
                
            except Exception as e:
                print(f"Error saving recording: {e}")

    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()

def main():
    recorder = LocalRecorder()
    
    print("\nMacBook Pro Microphone Recorder")
    print("1. Start Recording")
    print("2. List Available Devices")
    print("3. Exit")
    
    while True:
        choice = input("\nSelect option (1-3): ")
        
        if choice == "1":
            print("\nStarting recording...")
            if recorder.start_recording():
                recorder.stop_recording()
            break
            
        elif choice == "2":
            recorder.list_devices()
            
        elif choice == "3":
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please select 1-3.")

if __name__ == "__main__":
    main()