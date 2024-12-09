from pathlib import Path
import torch
import numpy as np
from typing import Dict, List, Optional, Set
import torchaudio
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeakerIdentifier:
    def __init__(self, speaker_manager, class_schedule):
        """Initialize SpeakerIdentifier with dependencies"""
        self.speaker_manager = speaker_manager
        self.class_schedule = class_schedule
        self.samples_dir = Path(__file__).parent.parent.parent / "speaker_samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.min_segment_duration = 1.0  # Minimum duration in seconds
        self.high_confidence_threshold = 0.7
        self.base_confidence_threshold = 0.45

    def extract_speaker_samples(self, audio: np.ndarray, segments: List[Dict], sample_rate: int = 16000) -> Dict[str, List[Dict]]:
        """Extract high-quality audio samples for each speaker"""
        speaker_samples = {}
        speaker_segments = {}
        
        # First collect all segments by speaker
        for segment in segments:
            if "speaker" not in segment:
                continue
                
            speaker = segment['speaker'].replace('SPEAKER_', '')
            speaker_id = f"SPEAKER_{speaker}"
            
            if speaker_id not in speaker_segments:
                speaker_segments[speaker_id] = []
            speaker_segments[speaker_id].append(segment)
        
        # Process each speaker's segments
        for speaker_id, speaker_segs in speaker_segments.items():
            speaker_samples[speaker_id] = []
            total_duration = 0
            
            # Sort segments by duration (longest first)
            speaker_segs.sort(key=lambda s: s['end'] - s['start'], reverse=True)
            
            for segment in speaker_segs:
                start_sample = int(segment["start"] * sample_rate)
                end_sample = int(segment["end"] * sample_rate)
                duration = end_sample - start_sample
                
                # Only accept segments longer than minimum duration
                if duration >= self.min_segment_duration * sample_rate:
                    segment_audio = audio[start_sample:end_sample]
                    speaker_samples[speaker_id].append({
                        "audio": segment_audio,
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"]
                    })
                    total_duration += duration
                    
                    # Stop if we have enough samples
                    if total_duration >= 10 * sample_rate and len(speaker_samples[speaker_id]) >= 2:
                        break
        
        # Remove speakers with no valid samples
        speaker_samples = {k: v for k, v in speaker_samples.items() if v}
        
        # Log extracted samples
        logger.info("\nExtracted samples for speakers:")
        for speaker_id, samples in speaker_samples.items():
            logger.info(f"\n{speaker_id} ({len(samples)} samples):")
            for sample in samples:
                logger.info(f"- Duration: {sample['end'] - sample['start']:.1f}s")
        
        return speaker_samples
    
    def save_speaker_samples(self, speaker_samples: Dict[str, List[Dict]], audio_path: Path) -> Optional[Path]:
        """Save speaker samples to disk"""
        if not speaker_samples:
            logger.info("\nNo new speakers to identify.")
            return None
            
        recording_name = audio_path.stem
        samples_dir = self.samples_dir / recording_name
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("\nSaving speaker samples...")
        
        for speaker_id, samples in speaker_samples.items():
            if not samples:
                continue
            
            # Create speaker directory
            speaker_dir = samples_dir / speaker_id
            speaker_dir.mkdir(exist_ok=True)
            logger.info(f"\nSaving samples for {speaker_id}:")
            
            # Save samples
            for i, sample in enumerate(samples):
                sample_path = speaker_dir / f"sample_{i+1}.wav"
                # Convert to torch tensor for saving
                waveform = torch.from_numpy(sample["audio"]).float()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                torchaudio.save(sample_path, waveform, 16000)
                logger.info(f"- Saved {sample_path.name}")
                
                # Save sample metadata
                meta_path = speaker_dir / f"sample_{i+1}.json"
                with open(meta_path, 'w') as f:
                    json.dump({
                        "start": sample["start"],
                        "end": sample["end"],
                        "text": sample["text"]
                    }, f, indent=2)
        
        return samples_dir
    
    def identify_speakers(self, samples_dir: Path) -> bool:
            """Create or enhance speaker profiles from samples"""
            if not self.speaker_manager:
                logger.error("Speaker manager not initialized.")
                return False
                
            if not samples_dir:
                return False
                
            logger.info("\nStarting speaker identification...")
            identified_any = False
            
            for speaker_dir in samples_dir.glob("SPEAKER_*"):
                speaker_id = speaker_dir.name
                samples = list(speaker_dir.glob("sample_*.wav"))
                if not samples:
                    continue
                    
                # Find best match among existing profiles
                best_match = None
                highest_similarity = -1
                
                for name, stored_embedding in self.speaker_manager.speaker_embeddings.items():
                    similarities = []
                    for sample_path in samples:
                        try:
                            waveform, _ = torchaudio.load(sample_path)
                            waveform = waveform.to(self.speaker_manager.device)
                            embedding = self.speaker_manager.speaker_model.encode_batch(waveform)
                            
                            similarity = torch.nn.functional.cosine_similarity(
                                embedding[0].view(1, -1),
                                stored_embedding.view(1, -1)
                            ).item()
                            similarities.append(similarity)
                        except Exception as e:
                            logger.error(f"Error processing sample: {e}")
                            continue
                    
                    if similarities:
                        avg_similarity = sorted(similarities)[len(similarities)//2]  # median
                        if avg_similarity > highest_similarity:
                            highest_similarity = avg_similarity
                            best_match = name

                # Handle profile creation/enhancement
                if best_match and highest_similarity > 0.6:  # High confidence match
                    logger.info(f"\nConfidently identified as {best_match} (confidence: {highest_similarity:.2f})")
                    try:
                        # Automatically enhance profile
                        for sample_path in samples:
                            waveform, _ = torchaudio.load(sample_path)
                            audio = waveform.numpy()[0]
                            segment = {
                                'start': 0,
                                'end': len(audio) / 16000,
                                'speaker_name': best_match
                            }
                            self.speaker_manager.enhance_speaker_profile(best_match, audio, [segment])
                        logger.info(f"Enhanced profile for {best_match}")
                        identified_any = True
                    except Exception as e:
                        logger.error(f"Error enhancing profile: {e}")
                
                elif best_match and highest_similarity > 0.45:  # Medium confidence match
                    logger.info(f"\nThis might be {best_match} (confidence: {highest_similarity:.2f})")
                    if input(f"Would you like to enhance {best_match}'s profile with these samples? (y/n): ").lower() == 'y':
                        try:
                            for sample_path in samples:
                                waveform, _ = torchaudio.load(sample_path)
                                audio = waveform.numpy()[0]
                                segment = {
                                    'start': 0,
                                    'end': len(audio) / 16000,
                                    'speaker_name': best_match
                                }
                                self.speaker_manager.enhance_speaker_profile(best_match, audio, [segment])
                            logger.info(f"Enhanced profile for {best_match}")
                            identified_any = True
                        except Exception as e:
                            logger.error(f"Error enhancing profile: {e}")
                
                else:  # No good match found
                    name = input("\nEnter speaker name (or 'skip' to skip): ").strip()
                    if name.lower() != 'skip' and name:
                        try:
                            for sample_path in samples:
                                waveform, _ = torchaudio.load(sample_path)
                                audio = waveform.numpy()[0]
                                self.speaker_manager.create_profile_from_segment(
                                    audio,
                                    start_time=0,
                                    end_time=len(audio) / 16000,
                                    speaker_name=name
                                )
                            logger.info(f"Created profile for {name}")
                            identified_any = True
                        except Exception as e:
                            logger.error(f"Error creating profile: {e}")
            
            if identified_any:
                logger.info("\nSpeaker identification completed successfully!")
            else:
                logger.info("\nNo new speakers were identified.")
            
            return identified_any
    
    def identify_speakers_in_segments(self, audio: np.ndarray, segments: List[Dict], recording_time: Optional[datetime] = None) -> Set[str]:
        """Identify speakers in audio segments using voice embeddings"""
        if not self.speaker_manager:
            return set()
        
        # Get class information for confidence adjustment
        class_info = None
        if recording_time and self.class_schedule:
            class_info = self.class_schedule.get_class_info(recording_time)
            if class_info:
                logger.info(f"\nDetected class: {class_info['name']}")
                logger.info(f"Expected speakers: {', '.join(sorted(class_info['speakers']))}")
        
        # Collect segments by speaker
        speaker_segments = {}
        for segment in segments:
            if "speaker" not in segment:
                continue
                
            speaker_id = f"SPEAKER_{segment['speaker']}"
            if speaker_id not in speaker_segments:
                speaker_segments[speaker_id] = []
            speaker_segments[speaker_id].append(segment)
        
        logger.info(f"\nAnalyzing {len(speaker_segments)} distinct speakers...")
        
        # Process each speaker
        for speaker_id, speaker_segments_list in speaker_segments.items():
            # Skip if already identified
            if all("speaker_name" in segment for segment in speaker_segments_list):
                speaker_name = speaker_segments_list[0]["speaker_name"]
                logger.info(f"\nSpeaker {speaker_id} already identified as {speaker_name}")
                continue
            
            # Get embeddings for all segments
            segment_embeddings = []
            for segment in speaker_segments_list:
                if "speaker_name" in segment:
                    continue
                    
                start_sample = int(segment["start"] * 16000)
                end_sample = int(segment["end"] * 16000)
                segment_audio = audio[start_sample:end_sample]
                
                # Skip short segments
                if len(segment_audio) < self.min_segment_duration * 16000:
                    continue
                    
                embedding = self.speaker_manager.get_embedding_for_segment(segment_audio)
                if embedding is not None:
                    segment_embeddings.append(embedding)
            
            if not segment_embeddings:
                logger.info(f"No valid embeddings found for {speaker_id}")
                continue
            
            # Compare with known speakers
            best_match = None
            highest_similarity = -1
            
            for name, stored_embedding in self.speaker_manager.speaker_embeddings.items():
                similarities = []
                for embedding in segment_embeddings:
                    try:
                        # Calculate cosine similarity with L2 normalization
                        emb1 = torch.nn.functional.normalize(embedding.unsqueeze(0), p=2, dim=1)
                        emb2 = torch.nn.functional.normalize(stored_embedding.unsqueeze(0), p=2, dim=1)
                        similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()
                        similarities.append(similarity)
                    except Exception as e:
                        logger.error(f"Error calculating similarity: {e}")
                        continue
                
                if similarities:
                    # Use median similarity
                    similarities.sort()
                    median_similarity = similarities[len(similarities)//2]
                    
                    # Apply role-based boost
                    boost = self.speaker_manager.get_similarity_boost(name, class_info)
                    adjusted_similarity = median_similarity * boost
                    
                    if adjusted_similarity > highest_similarity:
                        highest_similarity = adjusted_similarity
                        best_match = name

            # Get confidence threshold
            if best_match:
                threshold = self.speaker_manager.get_speaker_confidence_threshold(best_match, class_info)
                
                if highest_similarity > threshold:
                    logger.info(f"\nIdentified {speaker_id} as {best_match} (confidence: {highest_similarity:.2f})")
                    
                    # Update all segments from this speaker
                    for segment in speaker_segments_list:
                        if "speaker_name" not in segment:
                            segment["speaker_name"] = best_match
                            segment["confidence"] = highest_similarity
                    
                    # Enhance profile with high-confidence segments
                    high_quality_segments = []
                    for segment in speaker_segments_list:
                        start_sample = int(segment["start"] * 16000)
                        end_sample = int(segment["end"] * 16000)
                        duration = (end_sample - start_sample) / 16000
                        
                        if duration >= self.min_segment_duration and highest_similarity > self.high_confidence_threshold:
                            high_quality_segments.append(segment)
                    
                    if high_quality_segments:
                        try:
                            for segment in high_quality_segments:
                                start_sample = int(segment["start"] * 16000)
                                end_sample = int(segment["end"] * 16000)
                                segment_audio = audio[start_sample:end_sample]
                                self.speaker_manager.enhance_speaker_profile(best_match, segment_audio, [segment])
                            logger.info(f"Enhanced profile with {len(high_quality_segments)} high-quality segments")
                        except Exception as e:
                            logger.error(f"Error enhancing profile: {e}")
                else:
                    logger.info(f"\nCould not confidently identify {speaker_id} (best match: {best_match}, confidence: {highest_similarity:.2f})")
            else:
                logger.info(f"\nNo matching profile found for {speaker_id}")
        
        # Generate summary
        identified_speakers = {segment.get("speaker_name") for segment in segments if "speaker_name" in segment}
        unidentified_speakers = {f"SPEAKER_{segment['speaker']}" for segment in segments 
                               if "speaker" in segment and "speaker_name" not in segment}
        
        logger.info(f"\nSpeaker identification summary:")
        logger.info(f"- Identified speakers ({len(identified_speakers)}): {', '.join(sorted(identified_speakers)) if identified_speakers else 'None'}")
        logger.info(f"- Unidentified speakers ({len(unidentified_speakers)}): {', '.join(sorted(unidentified_speakers)) if unidentified_speakers else 'None'}")
        
        return unidentified_speakers