from pathlib import Path
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def transcribe_audio(audio_path):
    print(f"Starting transcription of: {audio_path}")
    
    # Setup device for M1 Mac
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.float16  # M1 can handle float16
    
    try:
        # Initialize model and processor with M1 optimizations
        model_id = "openai/whisper-large-v3-turbo"
        
        print("Loading model...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Create the pipeline with specific parameters for M1
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            chunk_length_s=30,  # Process in 30-second chunks
            return_timestamps=True
        )
        
        # Perform transcription
        print("Transcribing...")
        result = pipe(
            str(audio_path),
            generate_kwargs={"language": "english", "task": "transcribe"}
        )
        
        return result["text"]
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def main():
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
                transcribed_text = transcribe_audio(selected_file)
                if transcribed_text:
                    print("\nTranscription:")
                    print("-" * 50)
                    print(transcribed_text)
                    print("-" * 50)
            else:
                print("Transcription cancelled")
        else:
            print(f"Please select a number between 0 and {len(wav_files) - 1}")
    except ValueError:
        print("Please enter a valid number")

if __name__ == "__main__":
    main()