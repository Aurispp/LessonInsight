# Start of transcriber.py
import warnings
from pathlib import Path
import whisperx
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Union
import os
from dotenv import load_dotenv
import torchaudio
import json
import shutil
import logging

from .schedule import ClassManager
from .speaker_manager import SpeakerManager  # Add this line

# Set up logging
logger = logging.getLogger(__name__)

class LessonTranscriber:
    def __init__(self, hf_token: Optional[str] = None, class_manager: Optional[ClassManager] = None):
        """Initialize the transcriber with all components"""
        # Set base attributes
        self.base_dir = Path(__file__).parent.parent.parent
        
        # Load token
        self.hf_token = self._load_hf_token(hf_token)
        self.device, self.compute_type = self._setup_device()
        
        # Initialize directories
        self.speaker_samples_dir = self.base_dir / "speaker_samples"
        self.speaker_samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Use provided class manager or create new one
        self.class_manager = class_manager or ClassManager()
        
        # Initialize speaker manager with class manager for profile handling
        self.speaker_manager = SpeakerManager(class_manager=self.class_manager)
    def _load_hf_token(self, hf_token: Optional[str]) -> Optional[str]:
        """
        Load HuggingFace token from (in order):
        1) The `hf_token` argument
        2) A .env file in `self.base_dir`
        3) The environment variable HUGGINGFACE_TOKEN
        """
        if hf_token:
            return hf_token  # Use the one explicitly passed in

        # Then try loading from .env
        env_path = self.base_dir / '.env'
        if env_path.exists():
            load_dotenv(str(env_path))

        # Finally try the environment
        token = os.getenv('HUGGINGFACE_TOKEN')
        if token:
            print("Successfully loaded HuggingFace token")
            return token

        print("Warning: No HuggingFace token found")
        return None

    def _perform_diarization(self, audio: np.ndarray, class_info: Optional[Dict] = None) -> Dict:
        """Perform speaker diarization on audio"""
        if not self.hf_token:
            logger.warning("No HuggingFace token available for diarization")
            return {"segments": []}
            
        try:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=self.device
            )
            
            min_speakers = 2
            if class_info and 'students' in class_info:
                max_speakers = len(class_info['students']) + 1
                logger.info(f"Adjusting for class size: expecting up to {max_speakers} speakers")
            else:
                max_speakers = 6
                logger.info("Using default maximum speakers: 6")
            
            # Main diarization call
            diarize_df = diarize_model(
                audio,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            # Convert DataFrame to list of dicts for compatibility
            segments = []
            for idx, row in diarize_df.iterrows():
                # Convert timings to float to ensure compatibility
                start_time = float(row['start'])
                end_time = float(row['end'])
                speaker = str(row['speaker'])
                
                segments.append({
                    "start": start_time,
                    "end": end_time,
                    "speaker": speaker
                })
            
            # Sort segments by start time
            segments.sort(key=lambda x: x["start"])
            
            return {
                "segments": segments,
                "speakers": list(set(s["speaker"] for s in segments))
            }
                
        except Exception as e:
            logger.error(f"Diarization error: {str(e)}")
            logger.exception("Full traceback:")
            return {"segments": []}


    def _handle_unidentified_speakers(self, audio: np.ndarray, segments: List[Dict], class_info: Optional[Dict] = None) -> None:
        """Extract and process unidentified speakers"""
        print("\nProcessing unidentified speakers...")
        
        # Skip if all speakers are identified
        unidentified_segments = [s for s in segments if "speaker" in s and "speaker_name" not in s]
        if not unidentified_segments:
            logger.info("All speakers are already identified")
            return

        # Create temporary directory for audio samples
        temp_samples_dir = self.speaker_samples_dir / "temp_samples"
        temp_samples_dir.mkdir(parents=True, exist_ok=True)
        sample_paths = []
        
        try:
            # Group segments by speaker
            speaker_segments = {}
            for segment in unidentified_segments:
                speaker_id = segment["speaker"]
                if speaker_id not in speaker_segments:
                    speaker_segments[speaker_id] = []
                speaker_segments[speaker_id].append(segment)
                
            # First handle teacher identification if needed
            if not self.speaker_manager.has_teacher_profile():
                print("\nNo teacher profile found. Let's identify the teacher first.")
                if self._identify_teacher(audio, speaker_segments, temp_samples_dir):
                    # Update speaker segments to remove identified teacher
                    for segment in segments:
                        if segment.get("speaker_name") == "Teacher":
                            speaker_id = segment["speaker"]
                            if speaker_id in speaker_segments:
                                del speaker_segments[speaker_id]
            
            # Process remaining unidentified speakers
            for speaker_id, speaker_segs in speaker_segments.items():
                # Skip if this speaker was already identified
                if any(seg.get("speaker_name") for seg in speaker_segs):
                    continue
                    
                print(f"\nProcessing speaker {speaker_id}...")
                
                # Get longest segments first
                speaker_segs.sort(key=lambda s: s["end"] - s["start"], reverse=True)
                
                # Collect audio samples
                samples = []
                total_duration = 0
                target_duration = 10  # seconds
                
                for segment in speaker_segs[:5]:  # Limit to 5 samples
                    duration = segment["end"] - segment["start"]
                    if duration < 1.0:  # Skip very short segments
                        continue
                        
                    start_sample = int(segment["start"] * 16000)
                    end_sample = int(segment["end"] * 16000)
                    segment_audio = audio[start_sample:end_sample]
                    
                    sample_path = temp_samples_dir / f"{speaker_id}_sample_{len(samples)}.wav"
                    waveform = torch.from_numpy(segment_audio).float()
                    if waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0)
                    torchaudio.save(sample_path, waveform, 16000)
                    
                    samples.append({
                        'path': sample_path,
                        'text': segment.get('text', ''),
                        'duration': duration
                    })
                    sample_paths.append(sample_path)
                    total_duration += duration
                    
                    if total_duration >= target_duration:
                        break
                
                if not samples:
                    logger.warning(f"No suitable samples found for {speaker_id}")
                    continue
                
                print(f"\nSpeaker {speaker_id} samples:")
                for i, sample in enumerate(samples, 1):
                    print(f"\n{i}. Duration: {sample['duration']:.1f}s")
                    if sample['text']:
                        print(f"   Text: {sample['text']}")
                    print(f"   Audio sample: {sample['path']}")
                
                # Wait for user to listen to samples
                input("\nListen to the samples using your system's audio player, then press Enter to continue...")
                
                if class_info:
                    print("\nKnown students in this class:")
                    for i, student in enumerate(class_info['students'], 1):
                        profile = None
                        if student.get('profile_id'):
                            profile = self.speaker_manager.get_profile_info(student['profile_id'])
                        print(f"{i}. {student['name']}")
                        if profile:
                            print(f"   Has voice profile with {profile.get('sample_count', 0)} samples")
                    print(f"{len(class_info['students']) + 1}. Different student")
                    
                    while True:
                        try:
                            choice = int(input("\nSelect student number (or 0 to skip): ").strip())
                            if choice == 0:
                                break
                            elif choice == len(class_info['students']) + 1:
                                name = input("Enter student name: ").strip()
                                if name:
                                    if self.class_manager.add_student_to_class(class_info['id'], name):
                                        print(f"Added new student: {name}")
                                        profile_id = self.speaker_manager.create_profile(
                                            samples[0]['path'],
                                            student['name'],
                                            class_info['id'],
                                            segments  # Pass the segments here
                                        )
                                        if profile_id:
                                            # Update segments for this speaker
                                            for segment in segments:
                                                if segment.get("speaker") == speaker_id:
                                                    segment["speaker_name"] = name
                                                    segment["profile_id"] = profile_id
                                    else:
                                        print("Failed to add student")
                                    break
                            elif 0 < choice <= len(class_info['students']):
                                student = class_info['students'][choice - 1]
                                profile_id = self.speaker_manager.create_profile(samples[0]['path'], student['name'], class_info['id'])
                                if profile_id:
                                    # Update segments for this speaker
                                    for segment in segments:
                                        if segment.get("speaker") == speaker_id:
                                            segment["speaker_name"] = student['name']
                                            segment["profile_id"] = profile_id
                                break
                            else:
                                print("Invalid choice")
                        except ValueError:
                            print("Invalid input")
        
        finally:
            # Clean up temporary files
            for path in sample_paths:
                try:
                    if path.exists():
                        path.unlink()
                except Exception as e:
                    logger.warning(f"Could not delete temporary file {path}: {e}")
            
            try:
                if temp_samples_dir.exists():
                    shutil.rmtree(temp_samples_dir)
            except Exception as e:
                logger.warning(f"Could not delete temporary directory {temp_samples_dir}: {e}")


    def _merge_similar_speakers(self, segments: List[Dict], max_speakers: int) -> List[Dict]:
        """
        Example helper method that merges adjacent/overlapping segments belonging to the same speaker.
        """
        if not segments:
            return segments
        
        # Sort by start time
        segments.sort(key=lambda seg: seg["start"])
        
        merged_segments = []
        current_seg = None
        
        for seg in segments:
            if current_seg is None:
                current_seg = seg
                continue
            
            # If same speaker & overlapping (or short gap), extend the end time
            if seg["speaker"] == current_seg["speaker"] and seg["start"] <= current_seg["end"] + 0.5:
                current_seg["end"] = max(current_seg["end"], seg["end"])
            else:
                merged_segments.append(current_seg)
                current_seg = seg
        
        # Add the last one
        if current_seg:
            merged_segments.append(current_seg)
        
        # If more speakers remain than max_speakers, do further merges or just log a warning
        unique_speakers = {s["speaker"] for s in merged_segments}
        if len(unique_speakers) > max_speakers:
            logger.warning(f"Still have {len(unique_speakers)} speakers, max is {max_speakers}. Further merges needed.")
        
        return merged_segments

    def _identify_teacher(self, audio: np.ndarray, speaker_segments: Dict, temp_dir: Path) -> None:
        """Identify and create teacher profile"""
        print("\nLet's identify the teacher's voice.")
        print("I'll play samples from each speaker, and you tell me which one is the teacher.")
        
        for speaker_id, segments in speaker_segments.items():
            # Filter out any segments missing 'end' or 'start'
            safe_segments = [
                seg for seg in segments 
                if "start" in seg and "end" in seg and seg.get('end') > seg.get('start', 0)
            ]
            
            # Sort by duration (largest first)
            safe_segments.sort(key=lambda s: s['end'] - s['start'], reverse=True)
            
            samples = []
            total_duration = 0
            
            for segment in safe_segments:
                start_sample = int(segment["start"] * 16000)
                end_sample = int(segment["end"] * 16000)
                duration = (end_sample - start_sample) / 16000
                
                if duration >= 1.0:
                    segment_audio = audio[start_sample:end_sample]
                    sample_path = temp_dir / f"{speaker_id}_teacher_sample_{len(samples)}.wav"
                    waveform = torch.from_numpy(segment_audio).float()
                    if waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0)
                    torchaudio.save(sample_path, waveform, 16000)
                    
                    samples.append({
                        'path': sample_path,
                        'text': segment.get('text', ''),
                        'duration': duration
                    })
                    total_duration += duration
                    
                    if total_duration >= 10:  # Max 10 seconds
                        break
            
            if samples:
                print(f"\nSpeaker {speaker_id} samples:")
                for i, sample in enumerate(samples, 1):
                    print(f"\n{i}. Duration: {sample['duration']:.1f}s")
                    print(f"   Text: {sample['text']}")
                    print(f"   Audio sample: {sample['path']}")
                
                if input("\nIs this the teacher? (y/n): ").lower() == 'y':
                    if self.speaker_manager.create_teacher_profile(samples[0]['path']):
                        print("Teacher profile created successfully")
                        # Update segments
                        for seg in segments:
                            seg['speaker_name'] = 'Teacher'
                    else:
                        print("Failed to create teacher profile")
                    break


    def _create_profile(self, audio_path: Path, name: str, class_id: Optional[str], segments: List[Dict]) -> None:
        """Create or update speaker profile"""
        profile_id = self.speaker_manager.create_profile(audio_path, name, class_id)
        if profile_id:
            # Update segment information
            for segment in segments:
                segment['speaker_name'] = name
                segment['profile_id'] = profile_id
            print(f"Created/updated profile for {name}")
        else:
            print(f"Failed to create profile for {name}")

    def assign_speakers_to_words(self, diarize_segments: List[Dict], whisper_segments: List[Dict]) -> List[Dict]:
        """Custom implementation to assign speakers to word-level segments"""
        try:
            # Convert diarization segments to pandas DataFrame
            diarize_df = pd.DataFrame(diarize_segments)
            result_segments = []
            
            for segment in whisper_segments:
                if "words" not in segment:
                    continue
                    
                segment_speakers = []
                for word in segment["words"]:
                    word_start = float(word["start"])
                    word_end = float(word["end"])
                    
                    # Find overlapping speaker segments
                    overlaps = diarize_df[
                        (diarize_df["start"] <= word_end) & 
                        (diarize_df["end"] >= word_start)
                    ]
                    
                    if not overlaps.empty:
                        # Get the speaker with most overlap
                        intersections = np.minimum(overlaps["end"], word_end) - np.maximum(overlaps["start"], word_start)
                        speaker = overlaps.iloc[intersections.argmax()]["speaker"]
                        word["speaker"] = speaker
                        segment_speakers.append(speaker)
                    else:
                        word["speaker"] = "UNKNOWN"
                        segment_speakers.append("UNKNOWN")
                
                # Assign most common speaker to the segment
                if segment_speakers:
                    from collections import Counter
                    segment["speaker"] = Counter(segment_speakers).most_common(1)[0][0]
                
                result_segments.append(segment)
            
            return result_segments
        except Exception as e:
            logger.error(f"Error assigning speakers to words: {str(e)}")
            logger.exception("Full traceback:")
            return whisper_segments

    def transcribe_audio(self, audio_path: Union[str, Path], recording_time: Optional[datetime] = None,
                        class_info: Optional[Dict] = None) -> Optional[List[Dict]]:
        """Transcribe audio and identify speakers"""
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                print(f"Error: Audio file not found: {audio_path}")
                return None
                
            print(f"Starting transcription of: {audio_path}")
            
            # Get recording time and class info
            if recording_time is None:
                recording_time = datetime.fromtimestamp(audio_path.stat().st_ctime)
            
            if not class_info:
                class_info = self.class_manager.get_class_for_time(recording_time)
            
            if class_info:
                status = "during class" if class_info.get("is_during_class") else "near class time"
                print(f"\nDetected class: {class_info.get('name', '')}")
                schedule = class_info['matched_schedule']
                print(f"Time: {schedule['day']} {schedule['start_time']} to {schedule['end_time']}")
                if class_info.get('students'):
                    print(f"Students in class: {len(class_info['students'])}")
                    for student in class_info['students']:
                        print(f"- {student['name']}")
            
            # Load and transcribe audio
            audio = whisperx.load_audio(audio_path)
            model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type)
            
            result = model.transcribe(audio, batch_size=1)
            language = result["language"]
            print(f"Detected language: {language}")
            
            del model  # Free up memory
            
            # Align timestamps
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
                    diarize_result = self._perform_diarization(audio, class_info)
                    
                    if diarize_result["segments"]:
                        logger.info("Starting speaker processing...")
                        
                        # Use custom speaker assignment
                        result["segments"] = self.assign_speakers_to_words(
                            diarize_result["segments"],
                            result["segments"]
                        )
                        
                        # Try to match with existing profiles
                        if "segments" in result:
                            # First try automatic matching
                            result["segments"] = self.speaker_manager.match_speakers_with_profiles(
                                audio, 
                                result["segments"], 
                                class_info
                            )
                            
                            # Only handle remaining unmatched speakers
                            unmatched_segments = [
                                segment for segment in result["segments"]
                                if "speaker" in segment and "speaker_name" not in segment
                            ]
                            
                            if unmatched_segments:
                                logger.info(f"Found {len(unmatched_segments)} unmatched segments")
                                self._handle_unidentified_speakers(audio, result["segments"], class_info)
                            else:
                                logger.info("All speakers successfully matched")
                
                except Exception as e:
                    logger.error(f"Diarization error: {str(e)}")
                    logger.exception("Full traceback:")
                    print("Continuing with basic segments...")
            else:
                print("Skipping diarization (no HuggingFace token)")
            
            # Save transcription
            self._save_transcription(audio_path, result["segments"], class_info)
            
            # Display transcript
            self._display_transcript(result["segments"])
            
            return result["segments"]
            
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            logger.exception("Full traceback:")
            return None



    def _save_transcription(self, audio_path: Path, segments: List[Dict], class_info: Optional[Dict] = None):
        """Save transcription in both JSON and readable text formats"""
        transcript_base = audio_path.parent / audio_path.stem
        json_path = transcript_base.with_suffix('.json')
        text_path = transcript_base.with_suffix('.txt')
        
        # Prepare data for JSON
        transcription_data = {
            "audio_file": str(audio_path),
            "timestamp": datetime.now().isoformat(),
            "duration": segments[-1]["end"] if segments else 0,
            "segments": segments,
            "class_info": class_info
        }
        
        # Save JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_data, f, indent=2, ensure_ascii=False)
        
        # Prepare text transcript
        text_output = []
        current_speaker = None
        
        # Header
        text_output.append("Transcript of: " + audio_path.name)
        text_output.append("Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Class info block
        if class_info:
            text_output.append(f"Class: {class_info.get('name', '')}")
            if "matched_schedule" in class_info:
                schedule_day = class_info["matched_schedule"].get("day", "UnknownDay")
                start_t = class_info["matched_schedule"].get("start_time", "??:??")
                end_t = class_info["matched_schedule"].get("end_time", "??:??")
                text_output.append(f"Time: {schedule_day} {start_t} to {end_t}")
        
        text_output.append("-" * 80 + "\n")
        
        # Body of the transcript
        for segment in segments:
            timestamp = f"[{self._format_timestamp(segment['start'])} -> {self._format_timestamp(segment['end'])}]"
            speaker = segment.get('speaker_name', f"Speaker_{segment.get('speaker', 'Unknown')}")
            
            # Speaker change
            if speaker != current_speaker:
                text_output.append(f"\n{speaker}:")
                current_speaker = speaker
            
            text_output.append(f"{timestamp} {segment['text'].strip()}")
        
        # Save text file
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_output))
        
        print(f"\nTranscription saved to:")
        print(f"- JSON: {json_path}")
        print(f"- Text: {text_path}")


    def _display_transcript(self, segments: List[Dict]):
            """Display formatted transcript"""
            print("\nTranscription completed successfully!")
            print(f"Number of segments: {len(segments)}")
            print("\nFull Transcript:")
            print("=" * 80)
            
            current_speaker = None
            for segment in segments:
                speaker = segment.get('speaker_name', f"Speaker_{segment.get('speaker', 'Unknown')}")
                timestamp = f"[{self._format_timestamp(segment['start'])} -> {self._format_timestamp(segment['end'])}]"
                
                # Add speaker change marker
                if speaker != current_speaker:
                    print(f"\n{speaker}:")
                    current_speaker = speaker
                
                print(f"{timestamp} {segment['text'].strip()}")
            
            print("\n" + "=" * 80)

    def _format_timestamp(self, seconds: float) -> str:
            """Convert seconds to HH:MM:SS format"""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _identify_teacher(self, audio: np.ndarray, speaker_segments: Dict, temp_dir: Path) -> None:
        """Identify and create teacher profile"""
        print("\nLet's identify the teacher's voice.")
        print("I'll play samples from each speaker, and you tell me which one is the teacher.")
        
        for speaker_id, segments in speaker_segments.items():
            # Filter out any segments missing 'end' or 'start'
            safe_segments = [
                seg for seg in segments 
                if "start" in seg and "end" in seg and seg.get('end') > seg.get('start', 0)
            ]
            
            # Sort by duration (largest first)
            safe_segments.sort(key=lambda s: s['end'] - s['start'], reverse=True)
            
            samples = []
            total_duration = 0
            
            for segment in safe_segments:
                start_sample = int(segment["start"] * 16000)
                end_sample = int(segment["end"] * 16000)
                duration = (end_sample - start_sample) / 16000
                
                if duration >= 1.0:
                    segment_audio = audio[start_sample:end_sample]
                    sample_path = temp_dir / f"{speaker_id}_teacher_sample_{len(samples)}.wav"
                    waveform = torch.from_numpy(segment_audio).float()
                    if waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0)
                    torchaudio.save(sample_path, waveform, 16000)
                    
                    samples.append({
                        'path': sample_path,
                        'text': segment.get('text', ''),
                        'duration': duration
                    })
                    total_duration += duration
                    
                    if total_duration >= 10:  # Max 10 seconds
                        break
            
            if samples:
                print(f"\nSpeaker {speaker_id} samples:")
                for i, sample in enumerate(samples, 1):
                    print(f"\n{i}. Duration: {sample['duration']:.1f}s")
                    print(f"   Text: {sample['text']}")
                    print(f"   Audio sample: {sample['path']}")
                
                if input("\nIs this the teacher? (y/n): ").lower() == 'y':
                    if self.speaker_manager.create_teacher_profile(samples[0]['path']):
                        print("Teacher profile created successfully")
                        # Update segments
                        for seg in segments:
                            seg['speaker_name'] = 'Teacher'
                    else:
                        print("Failed to create teacher profile")
                    break


    def _load_hf_token(self, hf_token: Optional[str]) -> Optional[str]:
            """Load HuggingFace token from various sources"""
            # First try the passed token
            if hf_token:
                return hf_token
                
            # Then try loading from .env file
            env_path = self.base_dir / '.env'
            if env_path.exists():
                load_dotenv(str(env_path))
            
            # Get token from environment
            token = os.getenv('HUGGINGFACE_TOKEN')
            if token:
                print(f"Successfully loaded HuggingFace token")
                return token
                
            print("Warning: No HuggingFace token found")
            return None

    def _setup_device(self) -> tuple[str, str]:
            """Setup device and compute type"""
            if torch.cuda.is_available():
                return "cuda", "float16"
            return "cpu", "int8"