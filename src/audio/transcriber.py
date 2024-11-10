from pathlib import Path
import whisperx
import torch
import json
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class LessonTranscriber:
    def __init__(self, hf_token=None):
        """
        Initialize the transcriber optimized for M1 Mac
        Args:
            hf_token (str, optional): Your HuggingFace token for accessing diarization models
        """
        # Force CPU usage for M1 Mac compatibility
        self.device = "cpu"
        self.compute_type = "int8"  # Use int8 for better memory efficiency
        self.hf_token = hf_token
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe audio using WhisperX with settings optimized for M1 Mac
        """
        print(f"Starting transcription of: {audio_path}")
        
        try:
            # Load and transcribe audio with WhisperX
            print("Loading audio...")
            audio = whisperx.load_audio(audio_path)
            
            print("Loading model...")
            model = whisperx.load_model(
                "large-v2",
                self.device,
                compute_type=self.compute_type
            )
            
            print("Transcribing...")
            result = model.transcribe(
                audio,
                batch_size=1  # Smaller batch size for CPU
            )
            
            # Clean up memory
            del model
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            # Align whisper output
            print("Aligning timestamps...")
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=self.device
            )
            
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                self.device
            )
            
            # Clean up memory
            del model_a
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            # Perform diarization if token is provided
            if self.hf_token:
                print("Performing speaker diarization...")
                try:
                    diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=self.hf_token,
                        device=self.device
                    )
                    
                    diarize_segments = diarize_model(
                        audio,
                        min_speakers=2,
                        max_speakers=4
                    )
                    
                    # Assign speaker labels to words
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                except Exception as e:
                    print(f"Warning: Diarization failed: {e}")
                    print("Continuing with transcription only...")
            
            return result["segments"]
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None

def main():
    # Get token from environment variable
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("Warning: No HuggingFace token found in .env file")
        print("Continuing without speaker diarization...")
    
    # Initialize transcriber with token from environment
    transcriber = LessonTranscriber(hf_token=hf_token)
    
    # Set up paths
    current_file = Path(__file__)
    audio_files_dir = current_file.parent.parent.parent / "audio_files"
    wav_files = list(audio_files_dir.glob("*.wav"))
    
    if not wav_files:
        print("No WAV files found in the audio directory!")
        return
    
    print("\nAvailable recordings:")
    for index, file in enumerate(wav_files):
        print(f"- [{index}] {file.name}")
    
    try:
        selected = int(input("\nEnter the number of the file to transcribe: "))
        
        if 0 <= selected < len(wav_files):
            selected_file = wav_files[selected]
            print(f"\nYou selected: {selected_file.name}")
            print("Proceed with transcription? (y/n)")
            confirm = input()
            
            if confirm.lower() == 'y':
                transcribed_segments = transcriber.transcribe_audio(selected_file)
                if transcribed_segments:
                    print("\nTranscription:")
                    print("-" * 50)
                    for segment in transcribed_segments:
                        if "speaker" in segment:
                            print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['speaker']}: {segment['text']}")
                        else:
                            print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")
                    print("-" * 50)
            else:
                print("Transcription cancelled")
        else:
            print(f"Please select a number between 0 and {len(wav_files) - 1}")
    except ValueError:
        print("Please enter a valid number")

if __name__ == "__main__":
    main()