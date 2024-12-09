import os
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Optional

# Import from local audio package
from audio.transcriber import LessonTranscriber
from audio.speaker_manager import SpeakerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LessonInsightApp:
    def __init__(self):
        # Get HuggingFace token from environment
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not self.hf_token:
            logger.warning("No HuggingFace token found in environment")
        
        # Initialize components
        self.transcriber = LessonTranscriber(hf_token=self.hf_token)
        self.speaker_manager = self.transcriber.speaker_manager
        
        # Set up paths
        self.audio_dir = Path(__file__).parent.parent / "audio_files"
        
    def list_recordings(self) -> List[Path]:
        """List available recordings with details"""
        logger.info(f"\nLooking for recordings in: {self.audio_dir}")
        logger.info("\nAvailable recordings:")
        
        recordings = list(self.audio_dir.glob("*.wav"))
        recordings.sort(key=lambda x: x.stat().st_ctime, reverse=True)
        
        for i, recording in enumerate(recordings):
            size_mb = recording.stat().st_size / (1024 * 1024)
            created = datetime.fromtimestamp(recording.stat().st_ctime)
            print(f"{i}: {recording.name} ({size_mb:.1f}MB, {created.strftime('%A %H:%M')})")
        
        return recordings
    
    def manage_profiles(self):
        """Manage speaker profiles"""
        while True:
            print("\nProfile Management")
            print("1. List all profiles")
            print("2. Delete profile")
            print("3. View profile details")
            print("4. Back to main menu")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                profiles = self.speaker_manager.list_profiles()
                print("\nCurrent Profiles:")
                for i, profile in enumerate(profiles, 1):
                    print(f"{i}. {profile['name']}")
                    print(f"   Created: {profile['created_at']}")
                    print(f"   Last updated: {profile['updated_at']}")
                    print(f"   Sample count: {profile['sample_count']}")
                
            elif choice == "2":
                profiles = self.speaker_manager.list_profiles()
                print("\nSelect profile to delete:")
                for i, profile in enumerate(profiles, 1):
                    print(f"{i}. {profile['name']}")
                
                try:
                    idx = int(input("\nEnter number (or 0 to cancel): ").strip())
                    if 0 < idx <= len(profiles):
                        profile = profiles[idx-1]
                        confirm = input(f"Are you sure you want to delete {profile['name']}? (y/n): ")
                        if confirm.lower() == 'y':
                            if self.speaker_manager.delete_profile(profile['name']):
                                print(f"Deleted profile: {profile['name']}")
                            else:
                                print("Failed to delete profile")
                except ValueError:
                    print("Invalid input")
                
            elif choice == "3":
                profiles = self.speaker_manager.list_profiles()
                print("\nSelect profile to view:")
                for i, profile in enumerate(profiles, 1):
                    print(f"{i}. {profile['name']}")
                
                try:
                    idx = int(input("\nEnter number (or 0 to cancel): ").strip())
                    if 0 < idx <= len(profiles):
                        profile = profiles[idx-1]
                        print(f"\nProfile: {profile['name']}")
                        print(f"Created: {profile['created_at']}")
                        print(f"Last updated: {profile['updated_at']}")
                        print(f"Sample count: {profile['sample_count']}")
                except ValueError:
                    print("Invalid input")
                
            elif choice == "4":
                break
    
    def transcribe_recording(self, recording_path: Path):
        """Transcribe a single recording"""
        try:
            recording_time = datetime.fromtimestamp(recording_path.stat().st_ctime)
            self.transcriber.transcribe_audio(recording_path, recording_time)
        except Exception as e:
            logger.error(f"Error transcribing recording: {e}")
    
    def run(self):
        """Main application loop"""
        while True:
            print("\nLessonInsight Menu")
            print("1. Transcribe recordings")
            print("2. Manage speaker profiles")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                recordings = self.list_recordings()
                if not recordings:
                    print("No recordings found in audio_files directory")
                    continue
                
                while True:
                    try:
                        selection = input("\nEnter the number of the recording to transcribe (or -1 to return): ")
                        if selection == "-1":
                            break
                            
                        idx = int(selection)
                        if 0 <= idx < len(recordings):
                            self.transcribe_recording(recordings[idx])
                        else:
                            print("Invalid selection")
                    except ValueError:
                        print("Please enter a valid number")
                    except Exception as e:
                        logger.error(f"Error: {e}")
            
            elif choice == "2":
                self.manage_profiles()
            
            elif choice == "3":
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please try again.")

def main():
    app = LessonInsightApp()
    app.run()

if __name__ == "__main__":
    main()