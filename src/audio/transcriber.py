from pathlib import Path
import whisperx
import torch

def transcribe_audio(audio_path):
    """
    Transcribe audio using WhisperX with settings optimized for M1 Mac (CPU mode)
    """
    print(f"Starting transcription of: {audio_path}")
    
    try:
        # Force CPU usage
        device = "cpu"
        compute_type = "int8"  # Use int8 for better memory efficiency
        
        # Load and transcribe audio with WhisperX
        print("Loading audio...")
        audio = whisperx.load_audio(audio_path)
        
        print("Loading model...")
        model = whisperx.load_model(
            "large-v2", 
            device,
            compute_type=compute_type
        )
        
        print("Transcribing...")
        result = model.transcribe(
            audio, 
            batch_size=1  # Smaller batch size for CPU
        )
        
        # Align whisper output
        print("Aligning timestamps...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], 
            device=device
        )
        result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            audio, 
            device
        )
        
        return result["segments"]
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def main():
    # Keep your original file selection interface
    current_file = Path(__file__)
    audio_files_dir = current_file.parent.parent.parent / "audio_files"
    wav_files = list(audio_files_dir.glob("*.wav"))
    
    if not wav_files:
        print("No WAV files found in the audio directory!")
        return
        
    print("\nAvailable recordings: ")
    for index, file in enumerate(wav_files):
        print(f"- [{index}] {file.name}")
        
    selected = input("\nEnter the number of the file you want to transcribe: ")
    
    try:
        selected_index = int(selected)
        
        if 0 <= selected_index < len(wav_files):
            selected_file = wav_files[selected_index]
            print(f"\nYou selected: {selected_file.name}")
            print("Proceed with transcription? (y/n)")
            confirm = input()
            
            if confirm.lower() == 'y':
                transcribed_segments = transcribe_audio(selected_file)
                if transcribed_segments:
                    print("\nTranscription:")
                    print("-" * 50)
                    for segment in transcribed_segments:
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