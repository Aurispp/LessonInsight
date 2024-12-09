import warnings
from pathlib import Path
import whisperx
import torch
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Union
import os
from dotenv import load_dotenv
import torchaudio
import json

from .schedule import ClassSchedule
from .speaker_manager import SpeakerManager
from .speaker_identifier import SpeakerIdentifier
from .transcription_writer import TranscriptionWriter

# Filter out specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="whisperx")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

class LessonTranscriber:
    def __init__(self, hf_token: Optional[str] = None, teacher_name: str = "Auris"):
        """Initialize the transcriber with all components"""
        self.teacher_name = teacher_name
        self.hf_token = self._load_hf_token(hf_token)
        self.device, self.compute_type = self._setup_device()
        
        try:
            self.speaker_manager = SpeakerManager(teacher_name=self.teacher_name)
            self.class_schedule = ClassSchedule(teacher_name=self.teacher_name)
            self.speaker_identifier = SpeakerIdentifier(self.speaker_manager, self.class_schedule)
        except Exception as e:
            print(f"Warning: Initialization failed: {e}")
            self.speaker_manager = None
            self.class_schedule = None
            self.speaker_identifier = None

        # Create speaker samples directory
        self.speaker_samples_dir = Path(__file__).parent.parent.parent / "speaker_samples"
        self.speaker_samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize teacher profile
        self.teacher_embedding = None
        if self.speaker_manager and "Teacher" in self.speaker_manager.speaker_embeddings:
            self.teacher_embedding = self.speaker_manager.speaker_embeddings["Teacher"]
            print("Loaded teacher profile")

    def _load_hf_token(self, hf_token: Optional[str]) -> Optional[str]:
        """Load HuggingFace token from various sources"""
        if hf_token:
            return hf_token
            
        # Try environment variable
        token = os.getenv('HUGGINGFACE_TOKEN')
        if token:
            return token
            
        # Try .env file
        env_path = Path(__file__).parent.parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            return os.getenv('HUGGINGFACE_TOKEN')
            
        return None

    def _setup_device(self) -> tuple[str, str]:
        """Setup device and compute type"""
        if torch.cuda.is_available():
            return "cuda", "float16"
        return "cpu", "int8"

    def _perform_transcription(self, audio: np.ndarray) -> List[Dict]:
        """Perform initial transcription and alignment"""
        # Load and run ASR model
        print("Loading ASR model...")
        model = whisperx.load_model(
            "large-v2",
            self.device,
            compute_type=self.compute_type,
            language="en"
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
        return result["segments"]
    
    def _parse_time_input(self, time_str: str) -> float:
            """Parse time input in various formats to seconds"""
            try:
                # Check if it's already in seconds
                return float(time_str)
            except ValueError:
                try:
                    # Try to parse MM:SS or HH:MM:SS format
                    parts = time_str.strip().split(':')
                    if len(parts) == 2:  # MM:SS
                        minutes, seconds = map(float, parts)
                        return minutes * 60 + seconds
                    elif len(parts) == 3:  # HH:MM:SS
                        hours, minutes, seconds = map(float, parts)
                        return hours * 3600 + minutes * 60 + seconds
                    else:
                        raise ValueError("Invalid time format")
                except:
                    raise ValueError("Please enter time as seconds (e.g., '5.2') or as MM:SS (e.g., '1:05')")
                
    def _perform_diarization(self, audio: np.ndarray, segments: List[Dict], recording_time: datetime, audio_path: Path) -> List[Dict]:
            """Perform diarization and speaker identification"""
            print("Performing speaker diarization...")
            try:
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=self.hf_token,
                    device=self.device
                )
                
                # Set number of speakers based on class info
                min_speakers = 1
                max_speakers = 6
                class_info = None
                if self.class_schedule:
                    class_info = self.class_schedule.get_class_info(recording_time)
                    if class_info:
                        num_expected_speakers = len(class_info['speakers'])
                        max_speakers = max(num_expected_speakers + 1, 5)
                        print(f"Expected speakers: {', '.join(class_info['speakers'])}")
                
                try:
                    # Perform diarization
                    diarize_result = diarize_model(
                        audio,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers
                    )
                    print("Raw diarization complete, attempting to assign speakers...")
                    
                    try:
                        # Debug information
                        print(f"Number of diarization segments: {len(diarize_result)}")
                        print(f"Number of transcription segments: {len(segments)}")
                        
                        # Create a mapping of time ranges to speakers
                        diarize_mapping = []
                        for _, row in diarize_result.iterrows():
                            diarize_mapping.append({
                                'start': row['start'],
                                'end': row['end'],
                                'speaker': f"SPEAKER_{row['speaker'].split('_')[-1]}"
                            })
                        
                        # Assign speakers to segments
                        print("\nAssigning speakers to segments...")
                        for segment in segments:
                            segment_mid = (segment['start'] + segment['end']) / 2
                            matching_speakers = []
                            
                            # Find overlapping diarization segments
                            for diar in diarize_mapping:
                                if diar['start'] <= segment_mid <= diar['end']:
                                    matching_speakers.append(diar['speaker'])
                            
                            # Assign speaker
                            if matching_speakers:
                                from collections import Counter
                                counts = Counter(matching_speakers)
                                segment['speaker'] = counts.most_common(1)[0][0]
                            else:
                                # Find closest segment if no overlap
                                min_dist = float('inf')
                                closest_speaker = "SPEAKER_0"
                                for diar in diarize_mapping:
                                    dist = min(abs(segment_mid - diar['start']), abs(segment_mid - diar['end']))
                                    if dist < min_dist:
                                        min_dist = dist
                                        closest_speaker = diar['speaker']
                                segment['speaker'] = closest_speaker
                        
                        print("Speaker assignment completed successfully")
                        
                    except Exception as e:
                        print(f"Error during speaker assignment: {e}")
                        print("Continuing with basic segments...")
                        for segment in segments:
                            segment['speaker'] = 'SPEAKER_0'
                
                except Exception as e:
                    print(f"Error during diarization: {e}")
                    print("Continuing with basic segments...")
                    for segment in segments:
                        segment['speaker'] = 'SPEAKER_0'
                    return segments
                
                # Proceed with speaker identification
                if self.speaker_identifier:
                    print("\nAttempting to match speakers with existing profiles...")
                    try:
                        unidentified = self.speaker_identifier.identify_speakers_in_segments(
                            audio, segments, recording_time
                        )
                        
                        # Handle unidentified speakers
                        if unidentified:
                            print(f"\nFound {len(unidentified)} unidentified speakers:", ", ".join(unidentified))
                            speaker_samples = self.speaker_identifier.extract_speaker_samples(audio, segments)
                            if speaker_samples:
                                samples_dir = self.speaker_identifier.save_speaker_samples(speaker_samples, audio_path)
                                if samples_dir:
                                    if input("\nWould you like to identify new speakers now? (y/n): ").lower() == 'y':
                                        self.speaker_identifier.identify_speakers(samples_dir)
                                        print("\nUpdating speaker identification with new profiles...")
                                        self.speaker_identifier.identify_speakers_in_segments(audio, segments, recording_time)
                        else:
                            print("\nAll speakers have been identified.")
                    except Exception as e:
                        print(f"Error during speaker identification: {e}")
                        print("Continuing with basic speaker labels...")
                
                return segments
            
            except Exception as e:
                print(f"Diarization error: {e}")
                for segment in segments:
                    segment['speaker'] = 'SPEAKER_0'
                return segments

    def identify_teacher(self, audio: np.ndarray, segments: List[Dict]) -> Optional[str]:
        """Identify which speaker is likely the teacher based on patterns"""
        if not segments:
            return None
            
        speaker_patterns = {}
        for segment in segments:
            if "speaker" not in segment:
                continue
                
            speaker_id = f"SPEAKER_{segment['speaker']}"
            if speaker_id not in speaker_patterns:
                speaker_patterns[speaker_id] = {
                    'segments': [],
                    'teacher_indicators': 0
                }
            
            text = segment['text'].lower()
            # Patterns that indicate this might be the teacher
            if any(pattern in text for pattern in [
                "say something",
                "tell me",
                "excellent",
                "very good",
                "go ahead",
                "class",
                "tell us",
                "now",
                "okay"
            ]):
                speaker_patterns[speaker_id]['teacher_indicators'] += 1
            
            speaker_patterns[speaker_id]['segments'].append(segment)
        
        # Find the speaker with the most teacher indicators
        most_likely_teacher = None
        max_indicators = 0
        for speaker_id, data in speaker_patterns.items():
            if data['teacher_indicators'] > max_indicators:
                max_indicators = data['teacher_indicators']
                most_likely_teacher = speaker_id
        
        if most_likely_teacher and max_indicators >= 2:
            print(f"\nDetected likely teacher: {most_likely_teacher}")
            print("Teacher indicators found:")
            for segment in speaker_patterns[most_likely_teacher]['segments']:
                print(f"- {segment['text']}")
            return most_likely_teacher
        
        return None
    
    def create_teacher_profile(self, audio: np.ndarray, segments: List[Dict], teacher_id: str):
        """Create or update the teacher's profile"""
        if not self.speaker_manager:
            return
            
        # Collect all teacher segments
        teacher_segments = []
        for segment in segments:
            if "speaker" in segment and f"SPEAKER_{segment['speaker']}" == teacher_id:
                start_sample = int(segment["start"] * 16000)
                end_sample = int(segment["end"] * 16000)
                segment_audio = audio[start_sample:end_sample]
                
                # Only use segments longer than 1 second
                if len(segment_audio) >= 16000:
                    teacher_segments.append({
                        'audio': segment_audio,
                        'text': segment['text']
                    })
        
        if not teacher_segments:
            print("No suitable segments found for teacher profile")
            return
        
        # Use the longest segment for the profile
        longest_segment = max(teacher_segments, key=lambda x: len(x['audio']))
        
        try:
            print("\nCreating teacher profile...")
            print(f"Using sample: {longest_segment['text']}")
            
            # Convert numpy array to torch tensor
            waveform = torch.from_numpy(longest_segment['audio']).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # Save the audio sample
            sample_dir = self.speaker_samples_dir / "teacher"
            sample_dir.mkdir(exist_ok=True)
            sample_path = sample_dir / "teacher_sample.wav"
            torchaudio.save(sample_path, waveform, 16000)
            
            # Create the profile
            self.speaker_manager.create_profile(sample_path, "Teacher")
            self.teacher_embedding = self.speaker_manager.speaker_embeddings["Teacher"]
            print("Teacher profile created successfully")
            
            # Update all teacher segments
            for segment in segments:
                if "speaker" in segment and f"SPEAKER_{segment['speaker']}" == teacher_id:
                    segment["speaker_name"] = "Teacher"
                    segment["confidence"] = 1.0
            
        except Exception as e:
            print(f"Error creating teacher profile: {e}")
    
    def transcribe_audio(self, audio_path: Union[str, Path], recording_time: Optional[datetime] = None) -> Optional[List[Dict]]:
        """Transcribe audio and identify speakers"""
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                print(f"Error: Audio file not found: {audio_path}")
                return None
                
            print(f"Starting transcription of: {audio_path}")
            
            # Get recording time from file if not provided
            if recording_time is None:
                recording_time = datetime.fromtimestamp(audio_path.stat().st_ctime)
            
            # Load audio
            audio = whisperx.load_audio(audio_path)
            
            # Load and run ASR model
            print("Loading ASR model...")
            model = whisperx.load_model(
                "large-v2",
                self.device,
                compute_type=self.compute_type,
                language="en"
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
                    
                    # Set number of speakers based on class info
                    min_speakers = 1
                    max_speakers = 5
                    if self.class_schedule:
                        class_info = self.class_schedule.get_class_info(recording_time)
                        if class_info:
                            num_expected_speakers = len(class_info['speakers'])
                            max_speakers = max(num_expected_speakers + 1, 5)  # Add 1 for unexpected speakers
                    
                    diarize_segments = diarize_model(
                        audio,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers
                    )
                    
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    print("Diarization completed successfully")
                    
                    # First try to identify speakers using existing profiles
                    if self.speaker_manager:
                        print("\nAttempting to match speakers with existing profiles...")
                        unidentified_speakers = self.speaker_identifier.identify_speakers_in_segments(audio, result["segments"], recording_time)
                        
                        # Extract and save samples for unidentified speakers
                        if unidentified_speakers:
                            print(f"\nFound {len(unidentified_speakers)} unidentified speakers:", ", ".join(sorted(unidentified_speakers)))
                            speaker_samples = self.speaker_identifier.extract_speaker_samples(audio, result["segments"])
                            
                            if speaker_samples:
                                samples_dir = self.speaker_identifier.save_speaker_samples(speaker_samples, audio_path)
                                if samples_dir and input("\nWould you like to identify new speakers now? (y/n): ").lower() == 'y':
                                    # If any new profiles were created, re-run speaker identification
                                    if self.speaker_identifier.identify_speakers(samples_dir):
                                        print("\nUpdating speaker identification with new profiles...")
                                        self.speaker_identifier.identify_speakers_in_segments(audio, result["segments"], recording_time)
                        else:
                            print("\nAll speakers have been identified.")
                    
                except Exception as e:
                    print(f"Diarization error: {e}")
            else:
                print("Skipping diarization (no HuggingFace token)")
            
            # Save transcription
            try:
                self._save_transcription(audio_path, result["segments"])
            except Exception as e:
                print(f"Warning: Could not save transcription file: {e}")
            
            # Display transcript
            print("\nTranscription completed successfully!")
            print(f"Number of segments: {len(result['segments'])}")
            print("\nFull Transcript:")
            print("=" * 80)
            
            # Group segments by speaker
            current_speaker = None
            for segment in result["segments"]:
                speaker = segment.get('speaker_name', f"Speaker_{segment.get('speaker', 'Unknown')}")
                timestamp = f"[{self._format_timestamp(segment['start'])} -> {self._format_timestamp(segment['end'])}]"
                
                # Add speaker change marker
                if speaker != current_speaker:
                    print(f"\n{speaker}:")
                    current_speaker = speaker
                
                print(f"{timestamp} {segment['text'].strip()}")
            
            print("\n" + "=" * 80)
            
            return result["segments"]
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None
    
    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _save_transcription(self, audio_path: Path, segments: List[Dict]):
        """Save transcription in both JSON and readable text formats"""
        # Create base paths
        transcript_base = audio_path.parent / audio_path.stem
        json_path = transcript_base.with_suffix('.json')
        text_path = transcript_base.with_suffix('.txt')
        
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