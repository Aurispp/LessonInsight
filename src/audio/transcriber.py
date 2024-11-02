import whisper
import torch
import time
import os
from datetime import datetime

def get_latest_recording():
    audio_dir = "../audio_files"
    # Modified to look for both .wav and .m4a files
    files = [f for f in os.listdir(audio_dir) if f.endswith((".wav", ".m4a"))]
    if not files:
        return None
    # Sort files by timestamp (newest first)
    latest_file = sorted(files, reverse=True)[0]
    return os.path.join(audio_dir, latest_file)

def transcribe_audio(audio_file):
    try:
        if torch.backends.mps.is_available():
            print(f"MPS (Metal) is available but using CPU for stability")
            device = "cpu"
        else:
            device = "cpu"
        
        print(f"Processing file: {audio_file}")
        
        # Load model
        start_time = time.time()
        print("Loading model...")
        model = whisper.load_model("large-v3-turbo")  # Changed to large-v3-turbo
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Transcribe
        start_time = time.time()
        print("Transcribing...")
        result = model.transcribe(audio_file)
        transcribe_time = time.time() - start_time
        print(f"Transcription completed in {transcribe_time:.2f} seconds")
        
        return result["text"]
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    # Option 1: Use specific file
    audio_file = "../audio_files/Passeig Marítim, 370 5.m4a"
    
    # Option 2: Use latest recording (uncomment to use)
    # audio_file = get_latest_recording()
    
    if os.path.exists(audio_file):
        text = transcribe_audio(audio_file)
        if text:
            print("\nTranscription:", text)
    else:
        print(f"Audio file not found: {audio_file}")