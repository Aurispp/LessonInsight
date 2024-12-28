from datetime import datetime, time
from typing import Optional, Dict, List, Set
from pathlib import Path
import json
import logging
import os
import pytz

logger = logging.getLogger(__name__)

class ClassManager:
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the class manager with configuration storage"""
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent
            
        self.base_dir = base_dir
        self.config_dir = base_dir / "config"
        
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Config directory ensured at {self.config_dir}")
        except Exception as e:
            logger.error(f"Error creating config directory: {e}")
            raise
        
        self.class_config_file = self.config_dir / "class_config.json"
        self.profile_config_file = self.config_dir / "profile_config.json"
        
        self.timezone = pytz.timezone('Europe/Madrid')
        
        # Load configurations
        self.class_config = self._load_class_config()
        self.profile_config = self._load_profile_config()
        
        # Repair any issues with class config
        if not self.repair_class_config():
            logger.warning("Failed to repair class configuration")

    def _load_class_config(self) -> Dict:
        """Load or create class configuration"""
        try:
            if self.class_config_file.exists():
                with open(self.class_config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded class configuration with {len(config.get('classes', {}))} classes")
                return config
        except Exception as e:
            logger.error(f"Error loading class configuration: {e}")
        
        logger.info("Creating new class configuration")
        return {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "classes": {}
        }

    def _load_profile_config(self) -> Dict:
        """Load or create profile configuration"""
        try:
            if self.profile_config_file.exists():
                with open(self.profile_config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded profile configuration with {len(config.get('profile_classes', {}))} profiles")
                return config
        except Exception as e:
            logger.error(f"Error loading profile configuration: {e}")
            
        logger.info("Creating new profile configuration")
        return {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "profile_classes": {}
        }

    def _save_class_config(self) -> bool:
        """Save class configuration"""
        try:
            backup_path = None
            # Ensure directory exists
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Update timestamp
            self.class_config["last_updated"] = datetime.now().isoformat()
            
            # Create backup if file exists
            if self.class_config_file.exists():
                backup_path = self.class_config_file.with_suffix('.bak')
                self.class_config_file.rename(backup_path)
            
            # Save to temporary file first
            temp_path = self.class_config_file.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(self.class_config, f, indent=2)
            
            # Rename temp file to actual file
            if os.path.exists(self.class_config_file):
                os.remove(self.class_config_file)
            os.rename(temp_path, self.class_config_file)
            
            # Remove backup if save was successful
            if backup_path and backup_path.exists():
                backup_path.unlink()
                
            logger.info(f"Saved class configuration with {len(self.class_config['classes'])} classes")
            return True
            
        except Exception as e:
            logger.error(f"Error saving class configuration: {e}")
            # Restore backup if available
            if backup_path and backup_path.exists():
                backup_path.rename(self.class_config_file)
            return False

    def _save_profile_config(self) -> bool:
        """Save profile configuration"""
        try:
            backup_path = None
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Update timestamp
            self.profile_config["last_updated"] = datetime.now().isoformat()
            
            # Create backup if file exists
            if self.profile_config_file.exists():
                backup_path = self.profile_config_file.with_suffix('.bak')
                self.profile_config_file.rename(backup_path)
            
            # Save to temporary file first
            temp_path = self.profile_config_file.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(self.profile_config, f, indent=2)
            
            # Rename temp file to actual file
            if os.path.exists(self.profile_config_file):
                os.remove(self.profile_config_file)
            os.rename(temp_path, self.profile_config_file)
            
            # Remove backup if save was successful
            if backup_path and backup_path.exists():
                backup_path.unlink()
                
            logger.info(f"Saved profile configuration with {len(self.profile_config['profile_classes'])} profiles")
            return True
            
        except Exception as e:
            logger.error(f"Error saving profile configuration: {e}")
            # Restore backup if available
            if backup_path and backup_path.exists():
                backup_path.rename(self.profile_config_file)
            return False

    def add_class(self, name: str, schedule: List[Dict], max_students: int, 
                student_names: List[str] = None, description: str = "") -> Optional[str]:
        """Add a new class"""
        try:
            # Debug BEFORE any processing
            logger.info("=== Adding New Class ===")
            logger.info(f"Received parameters:")
            logger.info(f"Name (raw): '{name}'")
            logger.info(f"Name type: {type(name)}")
            logger.info(f"Schedule: {schedule}")
            logger.info(f"Max students: {max_students}")
            logger.info(f"Student names: {student_names}")
            logger.info(f"Description: '{description}'")
            
            # Validate name first
            if name is None:
                logger.error("Name is None")
                return None
                
            name = str(name).strip()
            logger.info(f"Name after strip: '{name}'")
            
            if not name:
                logger.error("Cannot create class without a name")
                return None
            
            # Check name uniqueness
            for class_id, class_info in self.class_config["classes"].items():
                current_name = class_info.get("name", "").strip().lower()
                logger.info(f"Comparing with existing class: '{current_name}'")
                if current_name == name.lower():
                    logger.error(f"Class name '{name}' already exists")
                    return None
            
            # Create class ID
            class_id = f"class_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Generated class ID: {class_id}")
            
            # Create class data
            timestamp = datetime.now().isoformat()
            class_data = {
                "name": name,  # Store the name here
                "description": description,
                "schedule": schedule,
                "max_students": max_students,
                "students": [],
                "created_at": timestamp,
                "updated_at": timestamp
            }
            
            # Debug the final class data
            logger.info(f"Final class data to be saved: {json.dumps(class_data, indent=2)}")
            
            # Add students if provided
            if student_names:
                for student_name in student_names[:max_students]:
                    if student_name.strip():
                        class_data["students"].append({
                            "name": student_name.strip(),
                            "profile_id": None,
                            "added_at": timestamp
                        })
            
            # Save to config
            self.class_config["classes"][class_id] = class_data
            self.class_config["last_updated"] = timestamp
            
            if self._save_class_config():
                logger.info(f"Successfully created class '{name}' with ID {class_id}")
                return class_id
                
            logger.error("Failed to save class configuration")
            return None
            
        except Exception as e:
            logger.error(f"Error creating class: {e}")
            logger.exception("Full traceback:")
            return None
        
    def repair_class_config(self) -> bool:
        """Repair and validate class configuration"""
        try:
            logger.info("Repairing class configuration...")
            fixed = False
            
            for class_id, class_info in self.class_config["classes"].items():
                # Remove unwanted fields
                if "profiles" in class_info:
                    del class_info["profiles"]
                    fixed = True
                    logger.info(f"Removed profiles array from class {class_id}")
                
                # Ensure required fields
                if "updated_at" not in class_info:
                    class_info["updated_at"] = class_info.get("created_at", datetime.now().isoformat())
                    fixed = True
                    logger.info(f"Added updated_at to class {class_id}")
                
                # Ensure non-empty name
                if "name" in class_info and not class_info["name"]:
                    logger.warning(f"Found class {class_id} with empty name")
                
                # Ensure proper student structure
                for student in class_info.get("students", []):
                    if "profile_id" not in student:
                        student["profile_id"] = None
                        fixed = True
                    if "added_at" not in student:
                        student["added_at"] = class_info.get("created_at")
                        fixed = True
            
            if fixed:
                return self._save_class_config()
            return True
            
        except Exception as e:
            logger.error(f"Error repairing class configuration: {e}")
            logger.exception("Full traceback:")
            return False
    
    def get_class_for_time(self, recording_time: datetime) -> Optional[Dict]:
        """Get class information based on recording time"""
        # Convert to local timezone if needed
        if recording_time.tzinfo is None:
            recording_time = self.timezone.localize(recording_time)
        else:
            recording_time = recording_time.astimezone(self.timezone)
            
        day = recording_time.strftime('%A')
        current_time = recording_time.time()
        
        # First try exact match
        matched_class = self._find_matching_class(day, current_time)
        if matched_class:
            return matched_class
        
        # If no match found, look for nearby classes (+/- 2 hours)
        nearby_classes = self._find_nearby_classes(recording_time)
        if nearby_classes:
            return nearby_classes[0]  # Return the closest match
            
        return None

    def _find_matching_class(self, day: str, current_time: time) -> Optional[Dict]:
        """Find exact class match for day and time"""
        for class_id, class_info in self.class_config["classes"].items():
            for schedule in class_info["schedule"]:
                if schedule["day"] == day:
                    try:
                        start_time = datetime.strptime(schedule["start_time"], '%H:%M').time()
                        end_time = datetime.strptime(schedule["end_time"], '%H:%M').time()
                        
                        # Check if time falls within class period
                        if start_time <= current_time <= end_time:
                            return {
                                "id": class_id,
                                **class_info,
                                "matched_schedule": schedule,
                                "is_during_class": True
                            }
                    except ValueError as e:
                        logger.error(f"Error parsing time for class {class_info.get('name', class_id)}: {e}")
                        continue
        return None

    def _find_nearby_classes(self, recording_time: datetime) -> List[Dict]:
        """Find classes within 2 hours of the recording time"""
        nearby_classes = []
        day = recording_time.strftime('%A')
        
        for class_id, class_info in self.class_config["classes"].items():
            for schedule in class_info["schedule"]:
                if schedule["day"] == day:
                    try:
                        start_time = datetime.strptime(schedule["start_time"], '%H:%M').time()
                        end_time = datetime.strptime(schedule["end_time"], '%H:%M').time()
                        
                        # Convert class times to datetime for comparison
                        class_start = datetime.combine(recording_time.date(), start_time)
                        class_end = datetime.combine(recording_time.date(), end_time)
                        class_start = self.timezone.localize(class_start)
                        class_end = self.timezone.localize(class_end)
                        
                        # Check if recording is within 2 hours of class
                        time_diff_start = abs((class_start - recording_time).total_seconds() / 3600)
                        time_diff_end = abs((class_end - recording_time).total_seconds() / 3600)
                        
                        if time_diff_start <= 2 or time_diff_end <= 2:
                            nearby_classes.append({
                                "id": class_id,
                                **class_info,
                                "matched_schedule": schedule,
                                "is_during_class": False,
                                "time_diff": min(time_diff_start, time_diff_end)
                            })
                    except ValueError as e:
                        logger.error(f"Error parsing time for class {class_info.get('name', class_id)}: {e}")
                        continue
        
        # Sort by time difference
        return sorted(nearby_classes, key=lambda x: x["time_diff"])

    def add_student_to_class(self, class_id: str, student_name: str) -> bool:
        """Add a student to a class"""
        if not class_id or not student_name:
            return False
            
        if class_id not in self.class_config["classes"]:
            logger.error(f"Class {class_id} not found")
            return False
            
        class_info = self.class_config["classes"][class_id]
        
        # Check if student already exists
        if any(s["name"].lower() == student_name.lower() for s in class_info["students"]):
            logger.error(f"Student {student_name} already exists in class {class_info.get('name', class_id)}")
            return False
            
        # Check max students limit
        if len(class_info["students"]) >= class_info["max_students"]:
            logger.error(f"Class {class_info.get('name', class_id)} is full")
            return False
            
        # Add student
        class_info["students"].append({
            "name": student_name,
            "profile_id": None,
            "added_at": datetime.now().isoformat()
        })
        
        class_info["updated_at"] = datetime.now().isoformat()
        return self._save_class_config()

    def update_student_name(self, class_id: str, old_name: str, new_name: str) -> bool:
        """Update a student's name"""
        if not class_id or not old_name or not new_name:
            return False
            
        if class_id not in self.class_config["classes"]:
            logger.error(f"Class {class_id} not found")
            return False
            
        class_info = self.class_config["classes"][class_id]
        
        # Check if new name already exists
        if any(s["name"].lower() == new_name.lower() for s in class_info["students"]):
            logger.error(f"Student name {new_name} already exists in class {class_info.get('name', class_id)}")
            return False
            
        # Find and update student
        for student in class_info["students"]:
            if student["name"].lower() == old_name.lower():
                student["name"] = new_name
                student["updated_at"] = datetime.now().isoformat()
                class_info["updated_at"] = datetime.now().isoformat()
                return self._save_class_config()
                
        logger.error(f"Student {old_name} not found in class {class_info.get('name', class_id)}")
        return False

    def update_class_size(self, class_id: str, new_size: int) -> bool:
        """Update maximum class size"""
        if not class_id or new_size <= 0:
            return False
            
        if class_id not in self.class_config["classes"]:
            logger.error(f"Class {class_id} not found")
            return False
            
        class_info = self.class_config["classes"][class_id]
        
        # Check if new size is less than current student count
        if new_size < len(class_info["students"]):
            logger.error(f"New size {new_size} is less than current student count {len(class_info['students'])}")
            return False
            
        class_info["max_students"] = new_size
        class_info["updated_at"] = datetime.now().isoformat()
        return self._save_class_config()

    def remove_student_from_class(self, class_id: str, student_name: str) -> bool:
        """Remove a student from a class"""
        if not class_id or not student_name:
            return False
            
        if class_id not in self.class_config["classes"]:
            logger.error(f"Class {class_id} not found")
            return False
            
        class_info = self.class_config["classes"][class_id]
        
        # Find and remove student
        for i, student in enumerate(class_info["students"]):
            if student["name"].lower() == student_name.lower():
                # Remove profile association if exists
                if student.get("profile_id"):
                    if student["profile_id"] in self.profile_config["profile_classes"]:
                        del self.profile_config["profile_classes"][student["profile_id"]]
                
                class_info["students"].pop(i)
                class_info["updated_at"] = datetime.now().isoformat()
                
                # Save both configurations
                if not self._save_class_config():
                    return False
                if not self._save_profile_config():
                    return False
                return True
                
        logger.error(f"Student {student_name} not found in class {class_info.get('name', class_id)}")
        return False

    def get_class(self, class_id: str) -> Optional[Dict]:
        """Get class information"""
        if not class_id:
            return None
            
        class_info = self.class_config["classes"].get(class_id)
        if class_info:
            return {"id": class_id, **class_info}
        return None

    def list_classes(self) -> List[Dict]:
        """Get all classes with their information"""
        return [
            {"id": class_id, **class_info}
            for class_id, class_info in self.class_config["classes"].items()
        ]

    def delete_class(self, class_id: str) -> bool:
        """Delete a class and its profile associations"""
        if not class_id:
            return False
            
        if class_id not in self.class_config["classes"]:
            logger.error(f"Class {class_id} not found")
            return False
        
        class_info = self.class_config["classes"][class_id]
        logger.info(f"Deleting class {class_info.get('name', class_id)}")
        
        # First save the profiles to remove
        profiles_to_remove = []
        for student in class_info.get("students", []):
            if student.get("profile_id"):
                profiles_to_remove.append(student["profile_id"])
        
        # Delete class
        del self.class_config["classes"][class_id]
        
        # Remove profile associations
        for profile_id in profiles_to_remove:
            if profile_id in self.profile_config["profile_classes"]:
                del self.profile_config["profile_classes"][profile_id]
        
        # Save both configurations
        success = True
        if not self._save_class_config():
            success = False
            logger.error("Failed to save class configuration")
        if not self._save_profile_config():
            success = False
            logger.error("Failed to save profile configuration")
        return success

    def get_profiles_for_class(self, class_id: str) -> List[Dict]:
        """Get all profiles associated with a class"""
        if not class_id or class_id not in self.class_config["classes"]:
            return []
            
        class_info = self.class_config["classes"][class_id]
        profiles = []
        
        for student in class_info["students"]:
            if student.get("profile_id"):
                profiles.append({
                    "student_name": student["name"],
                    "profile_id": student["profile_id"]
                })
        
        return profiles

    def get_student_info(self, class_id: str, student_name: str) -> Optional[Dict]:
        """Get student information including profile"""
        if not class_id or not student_name:
            return None
            
        class_info = self.get_class(class_id)
        if not class_info:
            return None
            
        for student in class_info["students"]:
            if student["name"].lower() == student_name.lower():
                return student
                
        return None

    def get_class_duration(self, class_id: str) -> Optional[Dict[str, List[int]]]:
        """Get class durations in minutes for each day"""
        class_info = self.get_class(class_id)
        if not class_info:
            return None
            
        durations = {}
        for session in class_info["schedule"]:
            try:
                start = datetime.strptime(session["start_time"], '%H:%M').time()
                end = datetime.strptime(session["end_time"], '%H:%M').time()
                
                # Calculate duration in minutes
                duration = (end.hour * 60 + end.minute) - (start.hour * 60 + start.minute)
                
                if session["day"] not in durations:
                    durations[session["day"]] = []
                durations[session["day"]].append(duration)
                
            except ValueError as e:
                logger.error(f"Error calculating duration for class {class_info.get('name', class_id)}: {e}")
                continue
                
        return durations if durations else None

    def validate_class_time(self, day: str, start_time: str, end_time: str) -> bool:
        """Validate class time format and values"""
        try:
            # Validate day
            valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            if day not in valid_days:
                logger.error(f"Invalid day: {day}")
                return False
            
            # Validate times
            start = datetime.strptime(start_time, '%H:%M').time()
            end = datetime.strptime(end_time, '%H:%M').time()
            
            if end <= start:
                logger.error(f"End time {end_time} must be after start time {start_time}")
                return False
            
            return True
            
        except ValueError as e:
            logger.error(f"Invalid time format: {e}")
            return False

    def get_class_by_name(self, name: str) -> Optional[Dict]:
        """Get class information by name"""
        if not name:
            return None
            
        for class_id, class_info in self.class_config["classes"].items():
            if class_info["name"].lower() == name.lower():
                return {"id": class_id, **class_info}
        
        return None