import torch
import torchaudio
from pathlib import Path
import pickle
from typing import Optional, Dict, Union, List, Tuple
from speechbrain.inference import SpeakerRecognition
import numpy as np
import warnings
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")

class SpeakerManager:
    def __init__(self, embeddings_path: Optional[Path] = None, teacher_name: str = "Teacher"):
        self.teacher_name = teacher_name
        self.embedding_size = 192
        self.device = "cpu"  # Force CPU for better compatibility
        self.min_sample_length = 16000  # 1 second at 16kHz
        self.target_length = 32000  # 2 seconds - target length for processed audio
        
        # Set paths
        if embeddings_path is None:
            embeddings_path = Path(__file__).parent / "speaker_embeddings.pkl"
        self.embeddings_path = embeddings_path
        self.backup_path = embeddings_path.with_suffix('.bak')
        
        # Initialize storage
        self.speaker_embeddings: Dict[str, torch.Tensor] = {}
        self.embedding_metadata: Dict[str, Dict] = {}
        
        try:
            logger.info("Loading speaker recognition model...")
            self.speaker_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
            
            self._load_embeddings()
            logger.info(f"Loaded {len(self.speaker_embeddings)} speaker profiles")
            
            if self.teacher_name not in self.speaker_embeddings:
                logger.warning(f"No profile found for teacher ({self.teacher_name})")
                
        except Exception as e:
            logger.error(f"Error initializing speaker recognition: {e}")
            raise

    def _process_audio(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Process audio to suitable format for embedding"""
        try:
            # Convert to tensor if needed
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()
            
            # Ensure correct shape
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            # Handle short segments
            if audio.shape[1] < self.min_sample_length:
                # Repeat the audio until it's long enough
                repeats = int(np.ceil(self.target_length / audio.shape[1]))
                audio = audio.repeat(1, repeats)
                # Trim to target length
                audio = audio[:, :self.target_length]
            elif audio.shape[1] > self.target_length:
                # Take center portion if too long
                start = (audio.shape[1] - self.target_length) // 2
                audio = audio[:, start:start + self.target_length]
            
            return audio.to(self.device)
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise

    def _validate_embedding(self, embedding: torch.Tensor) -> bool:
        """Validate embedding tensor"""
        try:
            if not isinstance(embedding, torch.Tensor):
                return False
            if embedding.dim() > 2:
                return False
            if embedding.shape[-1] != self.embedding_size:
                return False
            if torch.isnan(embedding).any() or torch.isinf(embedding).any():
                return False
            return True
        except Exception:
            return False

    def _normalize_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Normalize embedding"""
        try:
            if not self._validate_embedding(embedding):
                raise ValueError("Invalid embedding tensor")
            
            embedding = embedding.to(self.device).float()
            if embedding.dim() == 2:
                embedding = embedding.squeeze(0)
            
            # L2 normalization
            norm = torch.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                raise ValueError("Zero norm embedding")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding normalization failed: {e}")
            raise

    def _create_backup(self):
        """Create backup of embeddings file"""
        try:
            if self.embeddings_path.exists():
                import shutil
                shutil.copy2(self.embeddings_path, self.backup_path)
                logger.info("Created embeddings backup")
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")

    def _load_embeddings(self):
        """Load and validate embeddings"""
        try:
            if self.embeddings_path.exists():
                with open(self.embeddings_path, 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, dict):
                    if 'embeddings' in data and 'metadata' in data:
                        embeddings = data['embeddings']
                        self.embedding_metadata = data['metadata']
                    else:
                        embeddings = data
                        self.embedding_metadata = {name: {
                            'created_at': datetime.now().isoformat(),
                            'updated_at': datetime.now().isoformat(),
                            'sample_count': 1
                        } for name in embeddings.keys()}
                
                valid_embeddings = {}
                for name, emb in embeddings.items():
                    try:
                        if self._validate_embedding(emb):
                            valid_embeddings[name] = self._normalize_embedding(emb)
                        else:
                            logger.warning(f"Skipped invalid embedding for {name}")
                    except Exception as e:
                        logger.error(f"Error processing embedding for {name}: {e}")
                
                self.speaker_embeddings = valid_embeddings
                logger.info(f"Loaded valid embeddings for: {', '.join(valid_embeddings.keys())}")
                
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            self._restore_from_backup()

    def _restore_from_backup(self):
        """Restore from backup if available"""
        try:
            if self.backup_path.exists():
                logger.info("Attempting to restore from backup...")
                with open(self.backup_path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    self.speaker_embeddings = data
                    logger.info("Successfully restored from backup")
            else:
                logger.warning("No backup file found")
                self.speaker_embeddings = {}
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            self.speaker_embeddings = {}

    def save_embeddings(self):
        """Save embeddings with backup"""
        try:
            self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            self._create_backup()
            
            data = {
                'embeddings': self.speaker_embeddings,
                'metadata': self.embedding_metadata,
                'version': '2.0',
                'last_updated': datetime.now().isoformat()
            }
            
            temp_path = self.embeddings_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f)
            
            temp_path.replace(self.embeddings_path)
            logger.info(f"Saved {len(self.speaker_embeddings)} speaker profiles")
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            raise

    def get_embedding_for_segment(self, audio_segment: np.ndarray) -> Optional[torch.Tensor]:
        """Get embedding for an audio segment"""
        try:
            # Process audio
            audio_tensor = self._process_audio(audio_segment)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.speaker_model.encode_batch(audio_tensor)
                return self._normalize_embedding(embedding[0])
                
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None

    def create_profile_from_segment(
        self,
        audio: np.ndarray,
        start_time: float,
        end_time: float,
        speaker_name: str
    ):
        """Create a speaker profile from segment"""
        try:
            logger.info(f"Creating profile for {speaker_name} from segment")
            
            # Check if profile exists
            if speaker_name in self.speaker_embeddings:
                logger.info(f"Profile exists, enhancing instead")
                segment = {
                    'start': start_time,
                    'end': end_time,
                    'speaker_name': speaker_name
                }
                self.enhance_speaker_profile(speaker_name, audio, [segment])
                return
            
            # Process audio and get embedding
            audio_tensor = self._process_audio(audio)
            with torch.no_grad():
                embedding = self.speaker_model.encode_batch(audio_tensor)
                normalized_embedding = self._normalize_embedding(embedding[0])
            
            # Update storage
            self.speaker_embeddings[speaker_name] = normalized_embedding
            self.embedding_metadata[speaker_name] = {
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'sample_count': 1
            }
            
            self.save_embeddings()
            logger.info(f"Created profile for {speaker_name}")
            
        except Exception as e:
            logger.error(f"Error creating profile: {e}")
            raise

    def enhance_speaker_profile(self, speaker_name: str, audio: np.ndarray, segments: List[Dict]):
        """Enhance existing profile with new segments"""
        try:
            logger.info(f"Enhancing profile for {speaker_name}")
            embeddings = []
            
            # Get current embedding
            if speaker_name in self.speaker_embeddings:
                embeddings.append(self.speaker_embeddings[speaker_name])
            
            # Process new segments
            for segment in segments:
                if segment.get('speaker_name') == speaker_name:
                    start_sample = int(segment['start'] * 16000)
                    end_sample = int(segment['end'] * 16000)
                    segment_audio = audio[start_sample:end_sample]
                    
                    embedding = self.get_embedding_for_segment(segment_audio)
                    if embedding is not None:
                        embeddings.append(embedding)
            
            if embeddings:
                try:
                    # Weighted average
                    weights = torch.ones(len(embeddings), device=self.device)
                    if len(embeddings) > 1:
                        weights[0] = 2.0  # Higher weight for existing profile
                    weights = weights / weights.sum()
                    
                    stacked = torch.stack(embeddings)
                    combined = (stacked * weights.unsqueeze(1)).sum(dim=0)
                    normalized = self._normalize_embedding(combined)
                    
                    # Update storage
                    self.speaker_embeddings[speaker_name] = normalized
                    if speaker_name in self.embedding_metadata:
                        self.embedding_metadata[speaker_name].update({
                            'updated_at': datetime.now().isoformat(),
                            'sample_count': self.embedding_metadata[speaker_name].get('sample_count', 1) + len(embeddings) - 1
                        })
                    
                    self.save_embeddings()
                    logger.info(f"Enhanced profile with {len(embeddings)} segments")
                    
                except Exception as e:
                    logger.error(f"Error combining embeddings: {e}")
                    raise
            else:
                logger.warning("No valid segments for enhancement")
                
        except Exception as e:
            logger.error(f"Error enhancing profile: {e}")
            raise

    def get_speaker_confidence_threshold(self, speaker_name: str, class_info: Optional[Dict]) -> float:
        """Get dynamic confidence threshold"""
        base_threshold = 0.45
        
        if speaker_name == self.teacher_name:
            return base_threshold - 0.10
        
        if class_info and speaker_name in class_info['speakers']:
            return base_threshold - 0.05
        
        return base_threshold

    def get_similarity_boost(self, speaker_name: str, class_info: Optional[Dict]) -> float:
        """Get context-based similarity boost"""
        if speaker_name == self.teacher_name:
            return 1.2
        
        if class_info and speaker_name in class_info['speakers']:
            return 1.1
        
        return 1.0

    def is_teacher_profile(self, name: str) -> bool:
        """Check if profile is teacher"""
        return name == self.teacher_name

    def get_profile_info(self, speaker_name: str) -> Optional[Dict]:
        """Get profile metadata"""
        try:
            if speaker_name in self.embedding_metadata:
                info = self.embedding_metadata[speaker_name].copy()
                info['embedding_exists'] = speaker_name in self.speaker_embeddings
                return info
            return None
        except Exception as e:
            logger.error(f"Error getting profile info: {e}")
            return None

    def delete_profile(self, speaker_name: str) -> bool:
        """Delete a speaker profile"""
        try:
            if speaker_name in self.speaker_embeddings:
                del self.speaker_embeddings[speaker_name]
                if speaker_name in self.embedding_metadata:
                    del self.embedding_metadata[speaker_name]
                self.save_embeddings()
                logger.info(f"Deleted profile for {speaker_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting profile: {e}")
            return False

    def list_profiles(self) -> List[Dict]:
        """List all profiles"""
        try:
            profiles = []
            for name in self.speaker_embeddings.keys():
                info = self.get_profile_info(name)
                if info:
                    info['name'] = name
                    profiles.append(info)
            return profiles
        except Exception as e:
            logger.error(f"Error listing profiles: {e}")
            return []

    def cleanup_invalid_profiles(self) -> int:
        """Remove invalid profiles"""
        try:
            initial_count = len(self.speaker_embeddings)
            invalid_profiles = []
            
            for name, embedding in self.speaker_embeddings.items():
                if not self._validate_embedding(embedding):
                    invalid_profiles.append(name)
            
            for name in invalid_profiles:
                del self.speaker_embeddings[name]
                if name in self.embedding_metadata:
                    del self.embedding_metadata[name]
            
            if invalid_profiles:
                self.save_embeddings()
                logger.info(f"Removed {len(invalid_profiles)} invalid profiles")
            
            return len(invalid_profiles)
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0

    def create_profile(self, audio_path: Union[str, Path], speaker_name: str):
        """Create a speaker profile from an audio file"""
        try:
            logger.info(f"Creating profile for {speaker_name}")
            
            # Load and validate audio file
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Process audio
            audio_tensor = self._process_audio(waveform)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.speaker_model.encode_batch(audio_tensor)
                normalized_embedding = self._normalize_embedding(embedding[0])
            
            # Update storage
            self.speaker_embeddings[speaker_name] = normalized_embedding
            self.embedding_metadata[speaker_name] = {
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'sample_count': 1,
                'last_sample_path': str(audio_path)
            }
            
            self.save_embeddings()
            logger.info(f"Created profile for {speaker_name}")
            
        except Exception as e:
            logger.error(f"Error creating profile: {e}")
            raise