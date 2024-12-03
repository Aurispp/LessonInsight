import warnings
from pathlib import Path
import whisperx
import torch
import json
from datetime import datetime
import os
import torchaudio
from speechbrain.inference import SpeakerRecognition
import numpy as np
import pickle
from typing import Optional, Dict, List, Union
import threading
from dotenv import load_dotenv

# Filter out specific deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")

# Load environment variables
load_dotenv()

class SpeakerManager:
    def __init__(self, embeddings_path: Optional[Path] = None):
        """Initialize speaker recognition system"""
        if embeddings_path is None:
            embeddings_path = Path(__file__).parent / "speaker_embeddings.pkl"
        
        self.embeddings_path = embeddings_path
        print("Loading speaker recognition model...")
        
        # Determine device
        if torch.backends.mps.is_available():
            self.device = "mps"  # Use Metal Performance Shaders for M1
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        print(f"Using device: {self.device}")
        
        try:
            self.speaker_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
            
            self.speaker_embeddings = self._load_embeddings()
            print(f"Loaded {len(self.speaker_embeddings)} speaker profiles")
        except Exception as e:
            print(f"Error initializing speaker recognition: {e}")
            raise
    
    def _load_embeddings(self) -> Dict:
        """Load existing speaker embeddings"""
        if self.embeddings_path.exists():
            try:
                with open(self.embeddings_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading embeddings: {e}")
                return {}
        return {}
    
    def save_embeddings(self):
        """Save speaker embeddings to disk"""
        try:
            self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(self.speaker_embeddings, f)
            print(f"Saved {len(self.speaker_embeddings)} speaker profiles")
        except Exception as e:
            print(f"Error saving embeddings: {e}")

    def create_profile(self, audio_path: Union[str, Path], speaker_name: str):
        """Create a new speaker profile"""
        print(f"Creating profile for {speaker_name}...")
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Move to appropriate device
            waveform = waveform.to(self.device)
            
            # Get embedding
            embedding = self.speaker_model.encode_batch(waveform)
            self.speaker_embeddings[speaker_name] = embedding[0]  # Take first embedding
            self.save_embeddings()
            print(f"Created profile for {speaker_name}")
        except Exception as e:
            print(f"Error creating profile: {e}")
            raise

    def get_embedding_for_segment(self, audio_segment: np.ndarray) -> Optional[torch.Tensor]:
        """Get embedding for an audio segment"""
        try:
            # Convert numpy array to torch tensor if needed
            if isinstance(audio_segment, np.ndarray):
                audio_segment = torch.from_numpy(audio_segment).float()
            
            # Ensure audio is in the right format
            if not isinstance(audio_segment, torch.Tensor):
                raise ValueError("Audio segment must be numpy array or torch tensor")
            
            # Add batch dimension if needed
            if audio_segment.dim() == 1:
                audio_segment = audio_segment.unsqueeze(0)
            
            # Move to appropriate device
            audio_segment = audio_segment.to(self.device)
            
            # Get embedding
            embedding = self.speaker_model.encode_batch(audio_segment)
            return embedding[0]  # Return first embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

class LessonTranscriber:
    def __init__(self, hf_token: Optional[str] = None):
        """Initialize the transcriber with speaker recognition"""
        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"
        else:
            self.device = "cpu"
            self.compute_type = "int8"
            
        print(f"Using device: {self.device} with compute type: {self.compute_type}")
        
        # Load token with better error handling
        self.hf_token = hf_token
        if not self.hf_token:
            # Try to load from environment
            self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
            if not self.hf_token:
                # Try to load directly from .env file
                env_path = Path(__file__).parent.parent.parent / '.env'
                if env_path.exists():
                    print(f"Found .env file at: {env_path}")
                    try:
                        with open(env_path) as f:
                            for line in f:
                                if line.startswith('HUGGINGFACE_TOKEN='):
                                    self.hf_token = line.split('=')[1].strip()
                                    break
                    except Exception as e:
                        print(f"Error reading .env file: {e}")
                else:
                    print(f"No .env file found at: {env_path}")
            
        if not self.hf_token:
            print("Warning: No HuggingFace token found in arguments, environment, or .env file")
        else:
            print("HuggingFace token loaded successfully")
            
        try:
            self.speaker_manager = SpeakerManager()
        except Exception as e:
            print(f"Warning: Speaker recognition initialization failed: {e}")
            self.speaker_manager = None
    
    def transcribe_audio(self, audio_path: Union[str, Path]) -> Optional[List[Dict]]:
        """Transcribe audio and identify speakers"""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_path}")
            return None
            
        print(f"Starting transcription of: {audio_path}")
        
        try:
            # Load audio
            audio = whisperx.load_audio(audio_path)
            
            # Load and run ASR model
            print("Loading ASR model...")
            model = whisperx.load_model(
                "large-v2",
                self.device,
                compute_type=self.compute_type,
                language="en"  # You can make this configurable
            )
            
            result = model.transcribe(audio, batch_size=1)
            language = result["language"]
            print(f"Detected language: {language}")
            
            del model  # Free up memory
            
            # Align timestamps
            print("Aligning timestamps...")
            model_a, metadata = whisperx.load_align_model(
                language_code=language,
                device=self.device
            )
            
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                self.device,
                return_char_alignments=False
            )
            
            del model_a  # Free up memory
            
            # Perform diarization if token is available
            if self.hf_token:
                print("Performing speaker diarization...")
                try:
                    diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=self.hf_token,
                        device=self.device
                    )
                    
                    diarize_segments = diarize_model(
                        audio,
                        min_speakers=1,
                        max_speakers=5
                    )
                    
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    print("Diarization completed successfully")
                    
                    # Identify known speakers if speaker manager is available
                    if self.speaker_manager:
                        print("Matching with known speakers...")
                        self._identify_speakers(audio, result["segments"])
                    
                except Exception as e:
                    print(f"Diarization error: {e}")
            else:
                print("Skipping diarization (no HuggingFace token)")
            
            # Save transcription
            self._save_transcription(audio_path, result["segments"])
            
            return result["segments"]
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None

    def _identify_speakers(self, audio: np.ndarray, segments: List[Dict]):
        """Identify known speakers in segments"""
        for segment in segments:
            if "speaker" in segment:
                # Extract audio for this segment
                start_sample = int(segment["start"] * 16000)
                end_sample = int(segment["end"] * 16000)
                segment_audio = audio[start_sample:end_sample]
                
                # Skip very short segments
                if len(segment_audio) < 100:
                    continue
                
                try:
                    # Get embedding for this segment
                    embedding = self.speaker_manager.get_embedding_for_segment(segment_audio)
                    
                    if embedding is not None:
                        best_match = None
                        highest_similarity = -1
                        
                        # Compare with known speakers
                        for name, stored_embedding in self.speaker_manager.speaker_embeddings.items():
                            current_emb = embedding.view(1, -1)
                            stored_emb = stored_embedding.view(1, -1)
                            
                            # Calculate cosine similarity
                            similarity = torch.nn.functional.cosine_similarity(
                                current_emb, stored_emb
                            ).item()
                            
                            if similarity > highest_similarity:
                                highest_similarity = similarity
                                best_match = name
                        
                        # Assign speaker if confidence is high enough
                        if highest_similarity > 0.75:  # Confidence threshold
                            segment["speaker_name"] = best_match
                            segment["confidence"] = highest_similarity
                        else:
                            segment["speaker_name"] = f"Speaker_{segment['speaker']}"
                            segment["confidence"] = highest_similarity
                        
                except Exception as e:
                    print(f"Error processing segment: {e}")
                    segment["speaker_name"] = f"Speaker_{segment['speaker']}"
                    segment["confidence"] = 0.0

    def _save_transcription(self, audio_path: Path, segments: List[Dict]):
        """Save transcription in both JSON and readable text formats"""
        try:
            # Create base paths
            transcript_base = audio_path.parent / audio_path.stem
            json_path = transcript_base.with_suffix('_transcription.json')
            text_path = transcript_base.with_suffix('_transcript.txt')
            
            # Prepare data for JSON
            transcription_data = {
                "audio_file": str(audio_path),
                "timestamp": datetime.now().isoformat(),
                "duration": segments[-1]["end"] if segments else 0,
                "segments": segments
            }
            
            # Save JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(transcription_data, f, indent=2, ensure_ascii=False)
            
            # Create readable text transcript
            text_output = []
            current_speaker = None
            
            # Add header
            text_output.append("Transcript of: " + audio_path.name)
            text_output.append("Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            text_output.append("-" * 80 + "\n")
            
            # Process segments
            for segment in segments:
                timestamp = f"[{self._format_timestamp(segment['start'])} -> {self._format_timestamp(segment['end'])}]"
                speaker = segment.get('speaker_name', f"Speaker_{segment.get('speaker', 'Unknown')}")
                
                # Add speaker change marker
                if speaker != current_speaker:
                    text_output.append(f"\n{speaker}:")
                    current_speaker = speaker
                
                text_output.append(f"{timestamp} {segment['text'].strip()}")
            
            # Save text transcript
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(text_output))
            
            print(f"\nTranscription saved to:")
            print(f"- JSON: {json_path}")
            print(f"- Text: {text_path}")
            
            # Return the text transcript for display
            return '\n'.join(text_output)
            
        except Exception as e:
            print(f"Error saving transcription: {e}")
            return None
            
    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def main():
    """Main function for testing"""
    # Load environment variables with explicit path
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        print(f"Loading environment from: {env_path}")
        load_dotenv(env_path)
    else:
        print(f"No .env file found at: {env_path}")
    
    # Get HuggingFace token with debugging
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if hf_token:
        print("HuggingFace token found in environment")
        masked_token = hf_token[:4] + '*' * (len(hf_token) - 8) + hf_token[-4:]
        print(f"Token (masked): {masked_token}")
    else:
        print("No HuggingFace token found in environment variables")
    
    # Initialize transcriber
    transcriber = LessonTranscriber(hf_token=hf_token)
    
    # Get the audio file path
    audio_dir = Path(__file__).parent.parent.parent / "audio_files"
    print(f"Looking for recordings in: {audio_dir}")
    
    # List available recordings
    print("\nAvailable recordings:")
    recordings = list(audio_dir.glob("*.wav"))
    
    # Sort recordings by creation time (newest first)
    recordings.sort(key=lambda x: x.stat().st_ctime, reverse=True)
    
    for i, recording in enumerate(recordings):
        # Show file size and creation time
        size_mb = recording.stat().st_size / (1024 * 1024)
        created = datetime.fromtimestamp(recording.stat().st_ctime)
        print(f"{i}: {recording.name} ({size_mb:.1f}MB, created: {created})")
    
    if not recordings:
        print("No recordings found in audio_files directory")
        return
    
    # Get user selection
    try:
        selection = int(input("\nEnter the number of the recording to transcribe: "))
        if 0 <= selection < len(recordings):
            audio_path = recordings[selection]
            print(f"\nTranscribing {audio_path.name}...")
            segments = transcriber.transcribe_audio(audio_path)
            
            if segments:
                print("\nTranscription completed successfully!")
                print(f"Number of segments: {len(segments)}")
                print("\nFull Transcript:")
                print("=" * 80)
                
                # Group segments by speaker
                current_speaker = None
                for segment in segments:
                    speaker = segment.get('speaker_name', f"Speaker_{segment.get('speaker', 'Unknown')}")
                    timestamp = f"[{transcriber._format_timestamp(segment['start'])} -> {transcriber._format_timestamp(segment['end'])}]"
                    
                    # Add speaker change marker
                    if speaker != current_speaker:
                        print(f"\n{speaker}:")
                        current_speaker = speaker
                    
                    print(f"{timestamp} {segment['text'].strip()}")
                
                print("\n" + "=" * 80)
                print("\nTranscription files have been saved. You can find them in the same directory as the audio file.")
                
        else:
            print("Invalid selection")
    except ValueError:
        print("Please enter a valid number")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()