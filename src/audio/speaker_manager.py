import logging
from pathlib import Path
import torch
import numpy as np
from typing import Optional, Dict, List, Union
from datetime import datetime
import torchaudio
import pickle
import shutil
from speechbrain.pretrained import EncoderClassifier

logger = logging.getLogger(__name__)

class SpeakerManager:
    def __init__(self, class_manager=None):
        """Initialize speaker recognition system"""
        self.base_dir = Path(__file__).parent
        self.embeddings_file = self.base_dir / "speaker_embeddings.pkl"
        
        # Initialize storage
        self.profiles = {}  # Regular profiles
        self.embeddings = {}  # Regular embeddings
        self.teacher_profile = None  # Teacher profile
        self.teacher_embedding = None  # Teacher embedding
        self.class_manager = class_manager
        
        # Load speaker recognition model
        logger.info("Loading speaker recognition model...")
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        )
        
        # Load existing profiles
        self._load_profiles()
        logger.info(f"Loaded {len(self.profiles)} speaker profiles")


    def _validate_and_clean_segments(self, segments: List[Dict]) -> List[Dict]:
        """Validate and clean segments before processing"""
        if not segments:
            logger.warning("No segments provided for validation")
            return []
            
        valid_segments = []
        for i, segment in enumerate(segments):
            try:
                # Check required fields
                if "start" not in segment or "end" not in segment:
                    logger.warning(f"Segment {i} missing start/end time: {segment}")
                    continue
                    
                if "speaker" not in segment:
                    logger.warning(f"Segment {i} missing speaker: {segment}")
                    continue
                    
                # Validate timing
                start = float(segment["start"])
                end = float(segment["end"])
                
                if end <= start:
                    logger.warning(f"Segment {i} has invalid timing: start={start}, end={end}")
                    continue
                    
                if end - start < 0.1:  # Skip extremely short segments
                    logger.warning(f"Segment {i} too short ({end-start}s), skipping")
                    continue
                    
                valid_segments.append(segment)
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Error validating segment {i}: {e}")
                continue
                
        logger.info(f"Validated {len(valid_segments)}/{len(segments)} segments")
        return valid_segments

    def match_speakers_with_profiles(self, audio: np.ndarray, segments: List[Dict], class_info: Optional[Dict] = None) -> List[Dict]:
        """Match diarized speakers with known profiles"""
        # Validate segments first
        segments = self._validate_and_clean_segments(segments)
        if not segments:
            return segments

        try:
            # Track processed speakers to avoid duplicates
            processed_speakers = set()
            logger.info(f"Starting speaker matching process...")
            
            # Load and validate student profiles first
            if class_info and "students" in class_info:
                logger.info(f"Class info: {class_info}")
                for student in class_info["students"]:
                    # Try to find profile if not already attached
                    if not student.get("profile_id"):
                        profile_id = self.find_profile_by_name_and_class(student["name"], class_info["id"])
                        if profile_id:
                            student["profile_id"] = profile_id
                            logger.info(f"Found and attached profile {profile_id} to student {student['name']}")
            
            # First try to match teacher
            if self.has_teacher_profile():
                logger.info("Checking for teacher match...")
                logger.info(f"Teacher profile info: {self.teacher_profile}")
                teacher_embedding = self.get_teacher_embedding()
                if teacher_embedding is not None:
                    logger.info(f"Teacher embedding shape: {teacher_embedding.shape}, norm: {np.linalg.norm(teacher_embedding)}")
                
                best_match = None
                best_score = 0.40  # Threshold
                
                for speaker_id in {s.get("speaker") for s in segments}:
                    if speaker_id in processed_speakers:
                        continue
                        
                    # Get speaker segments
                    speaker_segments = [s for s in segments if s.get("speaker") == speaker_id]
                    speaker_audio = self._get_audio_for_speaker(audio, speaker_segments)
                    if speaker_audio is None:
                        logger.warning(f"Could not get audio for speaker {speaker_id}")
                        continue
                        
                    embedding = self.get_embedding_for_segment(speaker_audio)
                    if embedding is None:
                        logger.warning(f"Could not get embedding for speaker {speaker_id}")
                        continue
                    
                    logger.info(f"Speaker {speaker_id} embedding shape: {embedding.shape}, norm: {np.linalg.norm(embedding)}")
                    score = self.compare_embeddings(embedding, teacher_embedding)
                    logger.info(f"Teacher match score for {speaker_id}: {score}")
                    
                    if score > best_score:
                        best_score = score
                        best_match = speaker_id
                
                if best_match:
                    logger.info(f"Matched speaker {best_match} to teacher profile")
                    for segment in segments:
                        if segment.get("speaker") == best_match:
                            segment["speaker_name"] = "Teacher"
                            segment["profile_id"] = "teacher"
                    processed_speakers.add(best_match)
                else:
                    logger.info("No speakers matched teacher profile")
            
            # Then try to match students
            if class_info and "students" in class_info:
                logger.info(f"Attempting to match speakers with {len(class_info['students'])} known students")
                
                for student in class_info["students"]:
                    profile_id = student.get("profile_id")
                    logger.info(f"Checking student {student['name']}, profile_id: {profile_id}")
                    
                    if not profile_id or profile_id in processed_speakers:
                        logger.info(f"Skipping student {student['name']}: {'no profile' if not profile_id else 'already processed'}")
                        continue
                        
                    profile_embedding = self.get_profile_embedding(profile_id)
                    if profile_embedding is None:
                        logger.warning(f"No embedding found for student {student['name']}")
                        continue
                    
                    logger.info(f"Student {student['name']} profile embedding shape: {profile_embedding.shape}, norm: {np.linalg.norm(profile_embedding)}")
                    
                    # Get embeddings for unmatched speakers
                    unmatched_speakers = {}
                    for speaker in {s.get("speaker") for s in segments}:
                        if speaker and speaker not in processed_speakers:
                            speaker_segments = [s for s in segments if s.get("speaker") == speaker]
                            speaker_audio = self._get_audio_for_speaker(audio, speaker_segments)
                            if speaker_audio is not None:
                                embedding = self.get_embedding_for_segment(speaker_audio)
                                if embedding is not None:
                                    unmatched_speakers[speaker] = embedding
                                    logger.info(f"Got embedding for unmatched speaker {speaker}")
                    
                    # Find best match
                    best_match = None
                    best_score = 0.40  # Minimum threshold
                    
                    for speaker, embedding in unmatched_speakers.items():
                        score = self.compare_embeddings(embedding, profile_embedding)
                        logger.info(f"Match score for {student['name']} with {speaker}: {score}")
                        if score > best_score:
                            best_score = score
                            best_match = speaker
                    
                    if best_match:
                        logger.info(f"Matched {best_match} to {student['name']} (score: {best_score})")
                        for segment in segments:
                            if segment.get("speaker") == best_match:
                                segment["speaker_name"] = student["name"]
                                segment["profile_id"] = profile_id
                        processed_speakers.add(best_match)
                    else:
                        logger.info(f"No match found for student {student['name']}")
            
            return segments
            
        except Exception as e:
            logger.error(f"Error matching speakers: {e}")
            logger.exception("Full traceback:")
            return segments
        
    def _get_speaker_embedding(self, audio: np.ndarray, segments: List[Dict]) -> Optional[np.ndarray]:
        """Get embedding for a speaker from their segments"""
        try:
            if not segments:
                return None
                
            # Collect audio from segments
            total_duration = 0
            speaker_audio = []
            
            for segment in sorted(segments, key=lambda x: x["end"] - x["start"], reverse=True):
                duration = segment["end"] - segment["start"]
                if duration < 1.0:
                    continue
                    
                start_sample = int(segment["start"] * 16000)
                end_sample = int(segment["end"] * 16000)
                speaker_audio.extend(audio[start_sample:end_sample])
                total_duration += duration
                
                if total_duration >= 10:  # Cap at 10 seconds
                    break
            
            if speaker_audio:
                return self.get_embedding_for_segment(np.array(speaker_audio))
            return None
            
        except Exception as e:
            logger.error(f"Error getting speaker embedding: {e}")
            return None

    def _get_audio_for_speaker(self, audio: np.ndarray, segments: List[Dict]) -> Optional[np.ndarray]:
        """Extract audio for a speaker from their segments"""
        try:
            total_duration = 0
            speaker_audio = []
            
            # Sort segments by duration, longest first
            sorted_segs = sorted(segments, key=lambda x: x["end"] - x["start"], reverse=True)
            
            for seg in sorted_segs[:5]:  # Use up to 5 longest segments
                duration = seg["end"] - seg["start"]
                if total_duration >= 30:  # Cap at 30 seconds
                    break
                    
                start_sample = int(seg["start"] * 16000)
                end_sample = int(seg["end"] * 16000)
                speaker_audio.extend(audio[start_sample:end_sample])
                total_duration += duration
            
            return np.array(speaker_audio) if speaker_audio else None
            
        except Exception as e:
            logger.error(f"Error getting audio for speaker: {e}")
            return None
    
    def _load_profiles(self):
        """Load speaker profiles from disk"""
        try:
            if self.embeddings_file.exists():
                logger.info(f"Loading profiles from {self.embeddings_file}")
                with open(self.embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.profiles = data.get('profiles', {})
                    self.embeddings = data.get('embeddings', {})
                    self.teacher_profile = data.get('teacher_profile')
                    self.teacher_embedding = data.get('teacher_embedding')
                    
                    # Debug info
                    logger.info(f"Loaded {len(self.profiles)} student profiles")
                    for profile_id, profile in self.profiles.items():
                        embedding = self.embeddings.get(profile_id)
                        if embedding is not None:
                            logger.info(f"Profile {profile_id} ({profile.get('name')}): "
                                    f"samples={profile.get('sample_count')}, "
                                    f"embedding_shape={embedding.shape}, "
                                    f"embedding_norm={np.linalg.norm(embedding)}")
                    
                    if self.teacher_profile:
                        logger.info("Teacher profile loaded: "
                                f"samples={self.teacher_profile.get('sample_count')}, "
                                f"embedding_shape={self.teacher_embedding.shape if self.teacher_embedding is not None else 'None'}, "
                                f"embedding_norm={np.linalg.norm(self.teacher_embedding) if self.teacher_embedding is not None else 'None'}")
        except Exception as e:
            logger.error(f"Error loading speaker profiles: {e}")
            logger.exception("Full traceback:")
            self.profiles = {}
            self.embeddings = {}
            self.teacher_profile = None
            self.teacher_embedding = None

    def _save_profiles(self) -> bool:
        """Save speaker profiles to disk"""
        try:
            backup_path = None
            # Create backup if file exists
            if self.embeddings_file.exists():
                backup_path = self.embeddings_file.with_suffix('.bak')
                self.embeddings_file.rename(backup_path)
            
            # Save all profile data
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump({
                    'profiles': self.profiles,
                    'embeddings': self.embeddings,
                    'teacher_profile': self.teacher_profile,
                    'teacher_embedding': self.teacher_embedding
                }, f)
                
            # Remove backup if save was successful
            if backup_path and backup_path.exists():
                backup_path.unlink()
                
            logger.info(f"Saved {len(self.profiles)} speaker profiles")
            return True
            
        except Exception as e:
            logger.error(f"Error saving speaker profiles: {e}")
            # Restore backup if available
            if backup_path and backup_path.exists():
                backup_path.rename(self.embeddings_file)
            return False
        
    def _match_teacher(self, speaker_embeddings: Dict[str, np.ndarray]) -> Optional[str]:
        """Match speakers against teacher profile"""
        teacher_embedding = self.get_teacher_embedding()
        if teacher_embedding is None:
            return None
            
        best_match = None
        best_score = -1
        
        for speaker, embedding in list(speaker_embeddings.items()):
            score = self.compare_embeddings(embedding, teacher_embedding)
            logger.debug(f"Teacher match score for {speaker}: {score}")
            if score > best_score and score > 0.4:  # Threshold for matching
                best_score = score
                best_match = speaker
        
        if best_match:
            logger.info(f"Identified teacher as {best_match} with score {best_score}")
        
        return best_match
    def _match_students(self, speaker_embeddings: Dict[str, np.ndarray], 
                       class_info: Dict, segments: List[Dict]) -> None:
        """Match remaining speakers with student profiles"""
        for student in class_info["students"]:
            profile_id = student.get("profile_id")
            if not profile_id:
                continue
                
            profile_embedding = self.get_profile_embedding(profile_id)
            if profile_embedding is None:
                logger.warning(f"No embedding found for profile {profile_id}")
                continue
            
            best_match = None
            best_score = -1
            
            for speaker, embedding in list(speaker_embeddings.items()):
                score = self.compare_embeddings(embedding, profile_embedding)
                logger.debug(f"Student {student['name']} match score for {speaker}: {score}")
                if score > best_score and score > 0.40:  # Threshold for matching
                    best_score = score
                    best_match = speaker
            
            if best_match:
                logger.info(f"Identified student {student['name']} as {best_match} with score {best_score}")
                # Update segments with student match
                for segment in segments:
                    if segment.get("speaker") == best_match:
                        segment["speaker_name"] = student["name"]
                        segment["profile_id"] = profile_id
                # Remove matched speaker
                if best_match in speaker_embeddings:
                    del speaker_embeddings[best_match]
            else:
                logger.debug(f"No match found for student {student['name']}")
    
    def create_teacher_profile(self, audio_path: Union[str, Path]) -> bool:
        """Create or update teacher profile"""
        try:
            # Get embedding for audio
            new_embedding = self._get_embedding(audio_path)
            if new_embedding is None:
                return False
            
            if self.teacher_embedding is not None:
                # Update existing profile (weighted average)
                self.teacher_embedding = 0.7 * self.teacher_embedding + 0.3 * new_embedding
                self.teacher_embedding = self.teacher_embedding / np.linalg.norm(self.teacher_embedding)
                self.teacher_profile['sample_count'] += 1
            else:
                # Create new profile
                self.teacher_embedding = new_embedding
                self.teacher_profile = {
                    'name': 'Teacher',
                    'created_at': datetime.now().isoformat(),
                    'sample_count': 1
                }
            
            self.teacher_profile['updated_at'] = datetime.now().isoformat()
            return self._save_profiles()
            
        except Exception as e:
            logger.error(f"Error creating/updating teacher profile: {e}")
            return False

    def find_profile_by_name_and_class(self, name: str, class_id: Optional[str]) -> Optional[str]:
        """Find existing profile ID by student name and class ID (public version)"""
        if not name:
            return None
            
        # Search through profiles
        for profile_id, profile in self.profiles.items():
            logger.info(f"Checking profile {profile_id}: {profile}")
            if (profile.get('class_id') == class_id and 
                profile.get('name', '').lower() == name.lower()):
                logger.info(f"Found existing profile for {name} in class {class_id}")
                return profile_id
        
        logger.info(f"No existing profile found for {name} in class {class_id}")
        return None

    def create_profile(self, audio_path: Union[str, Path], name: str, class_id: str, segments: Optional[List[Dict]] = None) -> Optional[str]:
        """Create or update speaker profile"""
        logger.info(f"Creating/updating profile for {name} with audio from {audio_path}")
        try:
            # Try to find existing profile first
            profile_id = self.find_profile_by_name_and_class(name, class_id)
            logger.info(f"Profile lookup result: {profile_id}")
            
            if profile_id:
                # Update existing profile
                logger.info(f"Updating existing profile {profile_id}")
                success = self._enhance_profile(profile_id, audio_path)
                if not success:
                    logger.error("Failed to enhance existing profile")
                    return None
                
                # Update segments if provided
                if segments:
                    for segment in segments:
                        if segment.get("speaker") == name:
                            segment["speaker_name"] = name
                            segment["profile_id"] = profile_id
                    logger.info(f"Updated {len(segments)} segments with profile info")
                
                return profile_id

            # Create new profile
            profile_id = f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Creating new profile with ID: {profile_id}")
            
            # Get embedding
            embedding = self._get_embedding(audio_path)
            if embedding is None:
                logger.error("Failed to get embedding for new profile")
                return None
            
            # Create profile
            self.profiles[profile_id] = {
                'id': profile_id,
                'name': name,
                'class_id': class_id,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'sample_count': 1
            }
            
            # Store embedding
            self.embeddings[profile_id] = embedding
            
            # Update segments if provided
            if segments:
                for segment in segments:
                    if segment.get("speaker") == name:
                        segment["speaker_name"] = name
                        segment["profile_id"] = profile_id
                logger.info(f"Updated {len(segments)} segments with new profile info")
            
            # Save to disk
            if self._save_profiles():
                logger.info(f"Successfully created new profile for {name}")
                return profile_id
            
            logger.error("Failed to save profiles")
            return None
                
        except Exception as e:
            logger.error(f"Error creating profile: {e}")
            logger.exception("Full traceback:")
            return None

    def _enhance_profile(self, profile_id: str, audio_path: Union[str, Path]) -> bool:
        """Enhance existing profile with new audio sample"""
        try:
            if profile_id not in self.profiles:
                logger.error(f"Profile {profile_id} not found")
                return False

            # Get new embedding
            new_embedding = self._get_embedding(audio_path)
            if new_embedding is None:
                logger.error("Failed to get new embedding for enhancement")
                return False
            
            logger.info(f"New embedding shape: {new_embedding.shape}, norm: {np.linalg.norm(new_embedding)}")
            
            # Update embedding (weighted average)
            current_embedding = self.embeddings[profile_id]
            logger.info(f"Current embedding shape: {current_embedding.shape}, norm: {np.linalg.norm(current_embedding)}")
            
            sample_count = self.profiles[profile_id]['sample_count']
            profile_name = self.profiles[profile_id]['name']
            
            # More weight to existing profile if we have more samples
            weight_existing = min(0.8, 0.5 + (sample_count * 0.1))
            weight_new = 1 - weight_existing
            
            logger.info(f"Enhancing profile {profile_id} ({profile_name}): "
                    f"samples={sample_count}, weights={weight_existing:.2f}/{weight_new:.2f}")
            
            # Calculate weighted average
            updated_embedding = (current_embedding * weight_existing + new_embedding * weight_new)
            updated_embedding = updated_embedding / np.linalg.norm(updated_embedding)
            
            logger.info(f"Updated embedding shape: {updated_embedding.shape}, norm: {np.linalg.norm(updated_embedding)}")
            
            # Update profile
            self.embeddings[profile_id] = updated_embedding
            self.profiles[profile_id]['sample_count'] += 1
            self.profiles[profile_id]['updated_at'] = datetime.now().isoformat()
            
            # Save changes
            if not self._save_profiles():
                logger.error("Failed to save profile changes")
                return False
                
            logger.info(f"Enhanced profile for {profile_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error enhancing profile: {e}")
            logger.exception("Full traceback:")
            return False
        
    def _get_embedding(self, audio_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Get embedding for audio file"""
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Get embedding
            embedding = self.encoder.encode_batch(waveform)
            
            # Convert to numpy and normalize
            embedding = embedding.squeeze().detach().numpy()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None

    def get_embedding_for_segment(self, audio_segment: np.ndarray) -> Optional[np.ndarray]:
        """Get embedding for audio segment with error handling"""
        try:
            # Convert to float32 if needed
            if audio_segment.dtype != np.float32:
                audio_segment = audio_segment.astype(np.float32)
            
            # Convert to torch tensor
            waveform = torch.from_numpy(audio_segment).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.encoder.encode_batch(waveform)
            
            # Convert to numpy and normalize
            embedding = embedding.squeeze().detach().numpy()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting segment embedding: {e}")
            return None

    def get_profile_info(self, profile_id: str) -> Optional[Dict]:
        """Get profile information with caching"""
        if profile_id == "teacher":
            return self.teacher_profile
        return self.profiles.get(profile_id)

    def get_profile_embedding(self, profile_id: str) -> Optional[np.ndarray]:
        """Get embedding for a profile with caching"""
        if profile_id == "teacher":
            return self.teacher_embedding
        return self.embeddings.get(profile_id)

    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compare two embeddings using cosine similarity"""
        if emb1 is None or emb2 is None:
            logger.warning("Received None embedding in comparison")
            return 0.0
        try:
            # Ensure embeddings are normalized
            emb1_norm = emb1 / np.linalg.norm(emb1)
            emb2_norm = emb2 / np.linalg.norm(emb2)
            similarity = float(np.dot(emb1_norm, emb2_norm))
            logger.info(f"Embedding comparison score: {similarity}")
            return similarity
        except Exception as e:
            logger.error(f"Error comparing embeddings: {e}")
            return 0.0

    def has_teacher_profile(self) -> bool:
        """Check if teacher profile exists"""
        return self.teacher_profile is not None and self.teacher_embedding is not None

    def get_teacher_embedding(self) -> Optional[np.ndarray]:
        """Get teacher embedding"""
        return self.teacher_embedding


    def delete_profile(self, profile_id: str) -> bool:
        """Delete a speaker profile"""
        try:
            if profile_id == "teacher":
                self.teacher_profile = None
                self.teacher_embedding = None
            else:
                if profile_id in self.profiles:
                    del self.profiles[profile_id]
                if profile_id in self.embeddings:
                    del self.embeddings[profile_id]
                
            return self._save_profiles()
            
        except Exception as e:
            logger.error(f"Error deleting profile: {e}")
            return False

    def get_profiles_by_class(self, class_id: str) -> List[Dict]:
        """Get all profiles for a class"""
        return [
            {**profile, 'embedding': self.embeddings[profile_id]}
            for profile_id, profile in self.profiles.items()
            if profile.get('class_id') == class_id
        ]

    def clear_all_profiles(self) -> bool:
        """Clear all profiles (use with caution)"""
        try:
            self.profiles = {}
            self.embeddings = {}
            self.teacher_profile = None
            self.teacher_embedding = None
            return self._save_profiles()
        except Exception as e:
            logger.error(f"Error clearing profiles: {e}")
            return False