from pathlib import Path
from typing import List, Dict
from datetime import datetime
import json

class TranscriptionWriter:
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @classmethod
    def save_transcription(cls, audio_path: Path, segments: List[Dict]):
        """Save transcription in both JSON and readable text formats"""
        try:
            # Create base paths
            json_path = audio_path.with_name(f"{audio_path.stem}_transcription.json")
            text_path = audio_path.with_name(f"{audio_path.stem}_transcript.txt")
            
            # Save JSON
            transcription_data = {
                "audio_file": str(audio_path),
                "timestamp": datetime.now().isoformat(),
                "duration": segments[-1]["end"] if segments else 0,
                "segments": segments
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(transcription_data, f, indent=2, ensure_ascii=False)
            
            # Create and save text transcript
            cls.save_text_transcript(text_path, audio_path, segments)
            
            print(f"\nTranscription saved to:")
            print(f"- JSON: {json_path}")
            print(f"- Text: {text_path}")
            
        except Exception as e:
            print(f"Error saving transcription: {e}")

    @classmethod
    def save_text_transcript(cls, text_path: Path, audio_path: Path, segments: List[Dict]):
        """Save human-readable text transcript"""
        text_output = []
        current_speaker = None
        
        # Add header
        text_output.append("Transcript of: " + audio_path.name)
        text_output.append("Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        text_output.append("-" * 80 + "\n")
        
        # Process segments
        for segment in segments:
            timestamp = f"[{cls.format_timestamp(segment['start'])} -> {cls.format_timestamp(segment['end'])}]"
            speaker = segment.get('speaker_name', f"Speaker_{segment.get('speaker', 'Unknown')}")
            
            if speaker != current_speaker:
                text_output.append(f"\n{speaker}:")
                current_speaker = speaker
            
            text_output.append(f"{timestamp} {segment['text'].strip()}")
        
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_output))

    @classmethod
    def display_transcript(cls, segments: List[Dict]):
        """Display the transcript in the console"""
        print("\nFull Transcript:")
        print("=" * 80)
        
        current_speaker = None
        for segment in segments:
            speaker = segment.get('speaker_name', f"Speaker_{segment.get('speaker', 'Unknown')}")
            timestamp = f"[{cls.format_timestamp(segment['start'])} -> {cls.format_timestamp(segment['end'])}]"
            
            if speaker != current_speaker:
                print(f"\n{speaker}:")
                current_speaker = speaker
            
            print(f"{timestamp} {segment['text'].strip()}")
        
        print("\n" + "=" * 80)