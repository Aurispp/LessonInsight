import os
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Optional, Dict, Union
import pytz

# Import from local audio package
from audio.transcriber import LessonTranscriber
from audio.schedule import ClassManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filters=[
        lambda record: 'whisper result segments' not in record.msg.lower(),
        lambda record: 'diarize result' not in record.msg.lower(),
        lambda record: 'word_segments' not in record.msg.lower()
    ]
)

logger = logging.getLogger(__name__)

class LessonInsightApp:
    def __init__(self):
        # Initialize class manager first
        self.class_manager = ClassManager()
        
        # Get HuggingFace token from environment
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not self.hf_token:
            logger.warning("No HuggingFace token found in environment")
        
        # Initialize transcriber with existing class manager
        self.transcriber = LessonTranscriber(
            hf_token=self.hf_token,
            class_manager=self.class_manager
        )
        self.speaker_manager = self.transcriber.speaker_manager
        
        # Set up paths
        self.audio_dir = Path(__file__).parent.parent / "audio_files"
        self.audio_dir.mkdir(exist_ok=True)
        
        # Set timezone
        self.timezone = pytz.timezone('Europe/Madrid')
    
    def list_recordings(self) -> List[Path]:
        """List available recordings with details"""
        logger.info(f"Looking for recordings in: {self.audio_dir}")
        
        try:
            # Check if directory exists
            if not self.audio_dir.exists():
                logger.warning(f"Audio directory does not exist: {self.audio_dir}")
                self.audio_dir.mkdir(parents=True, exist_ok=True)
                return []
                
            # Get all WAV files
            recordings = list(self.audio_dir.glob("*.wav"))
            logger.info(f"Found {len(recordings)} WAV files")
            
            if not recordings:
                logger.info("No recordings found")
                return []
                
            # Sort by creation time
            recordings.sort(key=lambda x: x.stat().st_ctime, reverse=True)
            
            print("\nAvailable recordings:")
            for i, recording in enumerate(recordings):
                try:
                    size_mb = recording.stat().st_size / (1024 * 1024)
                    created = datetime.fromtimestamp(recording.stat().st_ctime)
                    created = self.timezone.localize(created)
                    
                    # Try to find matching class
                    class_info = self.class_manager.get_class_for_time(created)
                    class_str = ""
                    if class_info:
                        status = "during class" if class_info.get("is_during_class") else "near class time"
                        class_str = f" ({class_info['name']} - {status})"
                    
                    print(f"{i}: {recording.name} ({size_mb:.1f}MB, {created.strftime('%A %H:%M')}{class_str})")
                    
                except Exception as e:
                    logger.error(f"Error processing recording {recording}: {e}")
                    continue
            
            return recordings
            
        except Exception as e:
            logger.error(f"Error listing recordings: {e}")
            logger.exception("Full traceback:")
            return []

    def transcribe_recording(self, recording_path: Path):
        """Transcribe a single recording"""
        try:
            recording_time = datetime.fromtimestamp(recording_path.stat().st_ctime)
            recording_time = self.timezone.localize(recording_time)
            
            # Get class information
            class_info = self.class_manager.get_class_for_time(recording_time)
            
            # Show matched class and get confirmation
            if class_info:
                print(f"\nRecording time matches:")
                print(f"Class: {class_info['name']}")
                print(f"Time: {class_info['matched_schedule']['day']} {class_info['matched_schedule']['start_time']} to {class_info['matched_schedule']['end_time']}")
                print(f"Students:")
                for student in class_info['students']:
                    print(f"- {student['name']}")
                
                while True:
                    print("\nOptions:")
                    print("1. Proceed with this class")
                    print("2. Select a different class")
                    print("3. Return to menu")
                    
                    choice = input("\nEnter your choice (1-3): ").strip()
                    
                    if choice == "1":
                        break  # Proceed with current class_info
                    elif choice == "2":
                        class_info = self._assign_recording_to_class()
                        if not class_info:
                            return  # User cancelled or no class selected
                        break
                    elif choice == "3":
                        return
                    else:
                        print("Invalid choice. Please enter 1, 2, or 3.")
            else:
                print("\nNo matching class found for this recording time.")
                if input("Would you like to assign this recording to a class? (y/n): ").lower() == 'y':
                    class_info = self._assign_recording_to_class()
                    if not class_info:
                        return  # User cancelled or no class selected
                else:
                    return
            
            print(f"\nStarting transcription of: {recording_path}")
            
            if class_info:
                status = "during class" if class_info.get("is_during_class") else "near class time"
                print(f"\nDetected class: {class_info['name']}")
                print(f"Time: {class_info['matched_schedule']['day']} {class_info['matched_schedule']['start_time']} to {class_info['matched_schedule']['end_time']}")
                if class_info.get('students'):
                    print(f"Students in class: {len(class_info['students'])}")
                    for student in class_info['students']:
                        print(f"- {student['name']}")
            
            self.transcriber.transcribe_audio(recording_path, recording_time, class_info)
            
        except Exception as e:
            logger.error(f"Error transcribing recording: {e}")

    def _assign_recording_to_class(self) -> Optional[Dict]:
        """Let user assign a recording to a class"""
        classes = self.class_manager.list_classes()
        if not classes:
            print("No classes available. Please create a class first.")
            return None
        
        print("\nAvailable classes:")
        for i, class_info in enumerate(classes, 1):
            print(f"\n{i}. {class_info['name']}")
            for schedule in class_info['schedule']:
                print(f"   {schedule['day']}: {schedule['start_time']} to {schedule['end_time']}")
            print(f"   Students: {len(class_info.get('students', []))}/{class_info.get('max_students', 0)}")
            for student in class_info.get('students', []):
                print(f"   - {student['name']}")
        
        while True:
            try:
                choice = input("\nEnter class number (or 0 to cancel): ").strip()
                if choice == "0":
                    return None
                    
                idx = int(choice)
                if 0 < idx <= len(classes):
                    selected_class = classes[idx-1]
                    print(f"\nSelected: {selected_class['name']}")
                    print(f"Time: {selected_class['matched_schedule']['day']} {selected_class['matched_schedule']['start_time']} to {selected_class['matched_schedule']['end_time']}")
                    if input("Confirm selection? (y/n): ").lower() == 'y':
                        return selected_class
                else:
                    print(f"Please enter a number between 0 and {len(classes)}")
            except ValueError:
                print("Invalid input")
            
        return None

    def manage_classes(self):
        """Manage classes and profiles"""
        while True:
            print("\nClass Management")
            print("1. List all classes")
            print("2. Add new class")
            print("3. View/Edit class details")
            print("4. Delete class")
            print("5. Back to main menu")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                classes = self.class_manager.list_classes()
                if not classes:
                    print("\nNo classes found")
                    continue
                    
                print("\nCurrent Classes:")
                for i, class_info in enumerate(classes, 1):
                    print(f"\n{i}. {class_info['name']}")  # Make sure we display the name
                    for schedule in class_info['schedule']:
                        print(f"   {schedule['day']}: {schedule['start_time']} to {schedule['end_time']}")
                    if class_info.get('description'):
                        print(f"   Description: {class_info['description']}")
                    print(f"   Students: {len(class_info.get('students', []))}/{class_info.get('max_students', 0)}")
            
            elif choice == "2":
                name = input("\nEnter class name: ").strip()
                logger.info(f"User entered class name: '{name}'")  # Debug log

                if not name:
                    print("Name cannot be empty")
                    continue

                # Debug check before max students
                logger.info(f"Name before max students prompt: '{name}'")

                # Get max students
                while True:
                    try:
                        max_students = int(input("\nEnter maximum number of students: ").strip())
                        logger.info(f"User entered max students: {max_students}")  # Debug log
                        if max_students <= 0:
                            print("Number of students must be positive")
                            continue
                        break
                    except ValueError:
                        print("Please enter a valid number")
                
                # Debug check after max students
                logger.info(f"Name after max students prompt: '{name}'")

                # Get student names
                print("\nEnter student names (one per line)")
                print("Enter blank line when done")
                student_names = []
                while True:
                    student_name = input().strip()
                    if not student_name:
                        break
                    if len(student_names) < max_students:
                        student_names.append(student_name)
                    else:
                        print(f"Maximum {max_students} students allowed")
                        break
                
                # Debug check after student names
                logger.info(f"Name after student names: '{name}'")
                logger.info(f"Collected student names: {student_names}")

                print("\nEnter schedule (one entry per line)")
                print("Format: Day HH:MM HH:MM")
                print("Example: Monday 10:00 11:30")
                print("Enter blank line when done")
                
                schedule = []
                while True:
                    line = input().strip()
                    if not line:
                        break
                    
                    try:
                        parts = line.split()
                        if len(parts) != 3:
                            print("Invalid format. Use 'Day Start_Time End_Time'")
                            continue
                            
                        day, start_time, end_time = parts
                        
                        # Validate times
                        try:
                            start = datetime.strptime(start_time, '%H:%M').time()
                            end = datetime.strptime(end_time, '%H:%M').time()
                            if end <= start:
                                print(f"End time {end_time} must be after start time {start_time}")
                                continue
                        except ValueError:
                            print("Invalid time format. Use HH:MM (e.g., 09:30)")
                            continue
                            
                        schedule.append({
                            "day": day,
                            "start_time": start_time,
                            "end_time": end_time
                        })
                        print(f"Added {day} class: {start_time} to {end_time}")
                    except ValueError:
                        print("Invalid format. Use 'Day Start_Time End_Time'")
                        continue
                
                # Debug check before final creation
                logger.info(f"Name before description: '{name}'")
                
                description = input("\nEnter class description (optional): ").strip()
                
                # Final debug check before class creation
                logger.info(f"Final name before class creation: '{name}'")
                logger.info(f"Final values being passed to add_class:")
                logger.info(f"  - name: '{name}'")
                logger.info(f"  - schedule: {schedule}")
                logger.info(f"  - max_students: {max_students}")
                logger.info(f"  - student_names: {student_names}")
                logger.info(f"  - description: '{description}'")
                
                if schedule:
                    try:
                        class_id = self.class_manager.add_class(name, schedule, max_students, student_names, description)
                        if class_id:
                            print(f"\nCreated class: {name}")
                            if student_names:
                                print(f"Added {len(student_names)} students")
                        else:
                            print("\nFailed to create class")
                    except Exception as e:
                        logger.error(f"Error during class creation: {e}")
                        logger.exception("Full traceback")
                        print("\nAn error occurred while creating the class")
                else:
                    print("No schedule provided")
            
            elif choice == "3":
                classes = self.class_manager.list_classes()
                if not classes:
                    print("\nNo classes found")
                    continue
                
                print("\nSelect class to view/edit:")
                for i, class_info in enumerate(classes, 1):
                    print(f"{i}. {class_info['name']}")
                
                try:
                    idx = int(input("\nEnter number (or 0 to cancel): ").strip())
                    if 0 < idx <= len(classes):
                        class_info = classes[idx-1]
                        while True:
                            print(f"\nClass: {class_info['name']}")
                            print("\nSchedule:")
                            for schedule in class_info['schedule']:
                                print(f"- {schedule['day']}: {schedule['start_time']} to {schedule['end_time']}")
                            if class_info.get('description'):
                                print(f"\nDescription: {class_info['description']}")
                            
                            print(f"\nStudents ({len(class_info.get('students', []))}/{class_info.get('max_students', 0)}):")
                            for i, student in enumerate(class_info.get('students', []), 1):
                                print(f"{i}. {student['name']}")
                                if student.get('profile_id'):
                                    profile = self.speaker_manager.get_profile_info(student['profile_id'])
                                    if profile:
                                        print(f"   Voice profile: {profile.get('sample_count', 0)} samples")
                            
                            print("\nActions:")
                            print("1. Add student")
                            print("2. Remove student")
                            print("3. Edit student name")
                            print("4. Change max students")
                            print("5. Back")
                            
                            action = input("\nEnter action (1-5): ").strip()
                            
                            if action == "1":
                                if len(class_info['students']) >= class_info['max_students']:
                                    print("Class is full")
                                    continue
                                    
                                name = input("\nEnter student name: ").strip()
                                if name:
                                    if self.class_manager.add_student_to_class(class_info['id'], name):
                                        print(f"Added student: {name}")
                                        class_info = self.class_manager.get_class(class_info['id'])
                                    else:
                                        print("Failed to add student")
                            
                            elif action == "2":
                                if not class_info['students']:
                                    print("No students to remove")
                                    continue
                                    
                                name = input("\nEnter student name to remove: ").strip()
                                if name:
                                    if self.class_manager.remove_student_from_class(class_info['id'], name):
                                        print(f"Removed student: {name}")
                                        class_info = self.class_manager.get_class(class_info['id'])
                                    else:
                                        print("Student not found")
                            
                            elif action == "3":
                                if not class_info['students']:
                                    print("No students to edit")
                                    continue
                                    
                                old_name = input("\nEnter current student name: ").strip()
                                new_name = input("Enter new name: ").strip()
                                if old_name and new_name:
                                    if self.class_manager.update_student_name(class_info['id'], old_name, new_name):
                                        print(f"Updated student name: {old_name} -> {new_name}")
                                        class_info = self.class_manager.get_class(class_info['id'])
                                    else:
                                        print("Failed to update name")
                            
                            elif action == "4":
                                try:
                                    new_size = int(input("\nEnter new maximum students: ").strip())
                                    if new_size > 0:
                                        if self.class_manager.update_class_size(class_info['id'], new_size):
                                            print(f"Updated max students to: {new_size}")
                                            class_info = self.class_manager.get_class(class_info['id'])
                                        else:
                                            print("Failed to update class size")
                                    else:
                                        print("Number must be positive")
                                except ValueError:
                                    print("Please enter a valid number")
                            
                            elif action == "5":
                                break
                    
                except ValueError:
                    print("Invalid input")
            
            elif choice == "4":
                classes = self.class_manager.list_classes()
                if not classes:
                    print("\nNo classes found")
                    continue
                
                print("\nSelect class to delete:")
                for i, class_info in enumerate(classes, 1):
                    print(f"{i}. {class_info['name']}")
                
                try:
                    idx = int(input("\nEnter number (or 0 to cancel): ").strip())
                    if 0 < idx <= len(classes):
                        class_info = classes[idx-1]
                        if len(class_info.get('students', [])) > 0:
                            confirm = input(f"Class has {len(class_info['students'])} students. Delete anyway? (y/n): ").lower()
                            if confirm != 'y':
                                continue
                        
                        if self.class_manager.delete_class(class_info['id']):
                            print(f"Deleted class: {class_info['name']}")
                        else:
                            print("Failed to delete class")
                except ValueError:
                    print("Invalid input")
            
            elif choice == "5":
                break

    def run(self):
        """Main application loop"""
        while True:
            print("\nLessonInsight Menu")
            print("1. Transcribe recordings")
            print("2. Manage classes")
            print("3. Exit")
            
            try:
                choice = input("\nEnter your choice (1-3): ").strip()
                
                if choice == "1":
                    recordings = self.list_recordings()
                    if not recordings:
                        print("No recordings found in audio_files directory")
                        continue
                    
                    while True:
                        try:
                            selection = input("\nEnter the number of the recording to transcribe (or -1 to return): ").strip()
                            if selection == "-1":
                                break
                                
                            idx = int(selection)
                            if 0 <= idx < len(recordings):
                                self.transcribe_recording(recordings[idx])
                                break  # Exit after successful transcription
                            else:
                                print("Invalid selection. Please enter a number between 0 and", len(recordings)-1)
                        except ValueError:
                            print("Please enter a valid number")
                        except Exception as e:
                            logger.error(f"Error: {e}")
                            break  # Exit on error
                
                elif choice == "2":
                    self.manage_classes()
                
                elif choice == "3":
                    print("Goodbye!")
                    break
                
                else:
                    print("Invalid choice. Please enter a number between 1 and 3")
                    
            except Exception as e:
                logger.error(f"Error in main menu: {e}")
                logger.exception("Full traceback:")
                print("An error occurred. Please try again.")
                continue

def main():
    app = LessonInsightApp()
    app.run()

if __name__ == "__main__":
    main()