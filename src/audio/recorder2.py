import pyaudio
import wave
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple
import queue

class MultiSourceRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.format = pyaudio.paFloat32
        self.channels = 2
        self.rate = 48000
        self.chunk = 480
        self.mic_buffer = queue.Queue(maxsize=1000)
        self.system_buffer = queue.Queue(maxsize=1000)
        self.is_recording = False
        self.mic_stream: Optional[pyaudio.Stream] = None
        self.system_stream: Optional[pyaudio.Stream] = None
        self.recording_finished = threading.Event()
        self.save_finished = threading.Event()
        
        # Set up paths
        self.audio_dir = Path(__file__).parent.parent.parent / "audio_files"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def get_device_by_name(self, name: str) -> Optional[int]:
        """Get device ID by its name."""
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        for i in range(num_devices):
            device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
            if name.lower() in device_info.get('name').lower():
                return i
        return None

    def list_audio_devices(self):
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

    def setup_local_recording(self) -> bool:
        """Setup recording using MacBook's built-in microphone."""
        mic_id = self.get_device_by_name("MacBook Pro Microphone")
        if mic_id is None:
            print("Error: Could not find MacBook Pro Microphone")
            return False
        return self.setup_streams(mic_id=mic_id)

    def setup_online_recording(self) -> bool:
        """Setup recording using external microphone and BlackHole."""
        mic_id = self.get_device_by_name("External Microphone")
        blackhole_id = self.get_device_by_name("BlackHole")
        
        if mic_id is None:
            print("Error: Could not find External Microphone")
            return False
        if blackhole_id is None:
            print("Error: Could not find BlackHole device")
            return False
            
        return self.setup_streams(mic_id=mic_id, system_id=blackhole_id)

    def setup_streams(self, mic_id: Optional[int] = None, system_id: Optional[int] = None) -> bool:
        """Set up audio streams."""
        try:
            if mic_id is not None:
                self.mic_stream = self.audio.open(
                    format=self.format,
                    channels=1,  # Mono input
                    rate=self.rate,
                    input=True,
                    input_device_index=mic_id,
                    frames_per_buffer=self.chunk,
                    stream_callback=self._mic_callback
                )

            if system_id is not None:
                self.system_stream = self.audio.open(
                    format=self.format,
                    channels=2,  # Stereo input
                    rate=self.rate,
                    input=True,
                    input_device_index=system_id,
                    frames_per_buffer=self.chunk,
                    stream_callback=self._system_callback
                )
            return True
        except Exception as e:
            print(f"Error opening streams: {e}")
            return False

    def _mic_callback(self, in_data, frame_count, time_info, status) -> Tuple[bytes, int]:
        """Callback for microphone stream."""
        if self.is_recording:
            try:
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                stereo_data = np.repeat(audio_data, 2)
                self.mic_buffer.put_nowait(stereo_data)
            except queue.Full:
                pass  # Drop frame if buffer is full
        return (in_data, pyaudio.paContinue)

    def _system_callback(self, in_data, frame_count, time_info, status) -> Tuple[bytes, int]:
        """Callback for system audio stream."""
        if self.is_recording:
            try:
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                self.system_buffer.put_nowait(audio_data)
            except queue.Full:
                pass  # Drop frame if buffer is full
        return (in_data, pyaudio.paContinue)

    def _save_recording(self):
        """Mix and save the recorded audio."""
        mixed_frames = []
        
        while self.is_recording or not (self.mic_buffer.empty() and self.system_buffer.empty()):
            try:
                # Get frames from both sources
                try:
                    mic_data = self.mic_buffer.get(timeout=0.1) * 0.6
                except queue.Empty:
                    mic_data = np.zeros(self.chunk * 2, dtype=np.float32)
                
                try:
                    system_data = self.system_buffer.get(timeout=0.1) * 0.8
                except queue.Empty:
                    system_data = np.zeros(self.chunk * 2, dtype=np.float32)
                
                # Mix the frames
                mixed_frame = mic_data + system_data
                mixed_frames.append(mixed_frame)
                
            except Exception as e:
                print(f"Mixing error: {e}")
                continue

        # Save the recording
        if mixed_frames:
            try:
                # Concatenate and normalize
                audio_data = np.concatenate(mixed_frames)
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data / max_val * 0.9
                
                # Convert to int16
                audio_data = (audio_data * 32767).astype(np.int16)
                
                # Save to file
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = self.audio_dir / f"recording_{timestamp}.wav"
                
                with wave.open(str(filename), 'wb') as wave_file:
                    wave_file.setnchannels(2)
                    wave_file.setsampwidth(2)
                    wave_file.setframerate(self.rate)
                    wave_file.writeframes(audio_data.tobytes())
                
                print(f"\nRecording saved as: {filename}")
                
            except Exception as e:
                print(f"Error saving recording: {e}")

        self.save_finished.set()

    def start_recording(self) -> bool:
        """Start recording."""
        # Clear buffers
        while not self.mic_buffer.empty():
            self.mic_buffer.get()
        while not self.system_buffer.empty():
            self.system_buffer.get()
        
        self.recording_finished.clear()
        self.save_finished.clear()
        self.is_recording = True
        
        if self.mic_stream:
            self.mic_stream.start_stream()
        if self.system_stream:
            self.system_stream.start_stream()
        
        # Start saving thread
        threading.Thread(target=self._save_recording, daemon=True).start()
        
        # Start monitor thread
        threading.Thread(target=self._monitor_recording).start()
        
        print("\nRecording... Press Enter to stop")
        return True

    def _monitor_recording(self):
        """Monitor for stop signal."""
        input("")  # Wait for Enter key
        self.stop_recording()

    def stop_recording(self):
        """Stop the recording."""
        if not self.is_recording:
            return

        print("\nStopping recording...")
        self.is_recording = False
        
        # Wait for saving to complete
        self.save_finished.wait(timeout=10)
        
        if self.mic_stream:
            self.mic_stream.stop_stream()
            self.mic_stream.close()
        if self.system_stream:
            self.system_stream.stop_stream()
            self.system_stream.close()

    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'audio'):
            self.audio.terminate()

def main():
    recorder = MultiSourceRecorder()
    
    print("\nRecording Options:")
    print("1. Local Recording (MacBook Microphone only)")
    print("2. Online Recording (External Microphone + BlackHole)")
    print("3. List Available Devices")
    print("4. Exit")
    
    while True:
        choice = input("\nSelect option (1-4): ")
        
        if choice == "1":
            print("\nStarting local recording using MacBook microphone...")
            if recorder.setup_local_recording():
                recorder.start_recording()
            break
            
        elif choice == "2":
            print("\nStarting online recording setup...")
            print("\nSetup Instructions:")
            print("1. In macOS Audio MIDI Setup:")
            print("   - Create a Multi-Output Device")
            print("   - Select both your headphones/speakers AND BlackHole")
            print("   - Set this Multi-Output Device as your system output")
            print("2. In Zoom/Chrome:")
            print("   - Set the Multi-Output Device as your speaker")
            
            input("\nPress Enter when you've completed the setup...")
            
            if recorder.setup_online_recording():
                recorder.start_recording()
            break
            
        elif choice == "3":
            recorder.list_audio_devices()
            
        elif choice == "4":
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()