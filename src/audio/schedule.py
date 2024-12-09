from datetime import datetime, time
from typing import Optional, Dict, Set

class ClassSchedule:
    def __init__(self, teacher_name: str = "Auris"):
        """Initialize schedule with a default teacher that's added to all classes"""
        self.teacher_name = teacher_name
        self._base_schedule = {
            'Monday': {
                time(10, 0): {'name': 'Corever', 'speakers': {'Ivan', 'Raquel'}},
                time(12, 30): {'name': 'Lipotec', 'speakers': {'Maria', 'Juan'}},
                time(14, 15): {'name': 'Lubrizol', 'speakers': {'Cugat', 'Carlos', 'David'}},
                time(16, 0): {'name': 'Evening Class', 'speakers': {'Evelyn'}},
                time(17, 30): {'name': 'Evening Class', 'speakers': {'Pau'}},
                time(20, 0): {'name': 'Evening Class', 'speakers': {'Oriol'}},
            },
            'Tuesday': {
                time(8, 0): {'name': 'Morning Class', 'speakers': {'Andrea', 'Marga', 'Pedro'}},
                time(14, 0): {'name': 'GME', 'speakers': {'Joan', 'Pilar', 'Carolina'}},
                time(17, 15): {'name': 'Evening Class', 'speakers': {'Alex'}},
            },
            'Wednesday': {
                time(17, 30): {'name': 'Evening Class', 'speakers': {'Pau'}},
            },
            'Thursday': {
                time(8, 0): {'name': 'Corever', 'speakers': {'Ivan', 'Raquel'}},
                time(12, 30): {'name': 'Lipotec', 'speakers': {'Aida', 'Marti', 'Josep', 'Alex'}},
                time(14, 0): {'name': 'GME', 'speakers': {'Joan', 'Pilar', 'Carolina'}},
                time(16, 0): {'name': 'Evening Class', 'speakers': {'Claudia', 'Marti'}},
                time(18, 0): {'name': 'Evening Class', 'speakers': {'Oriol'}},
            },
            'Friday': {
                time(8, 0): {'name': 'Lipotec', 'speakers': {'Milagros', 'Ana', 'David', 'Cristina'}},
            }
        }

    @property
    def schedule(self) -> Dict:
        """Return schedule with teacher automatically added to all classes"""
        complete_schedule = {}
        for day, day_schedule in self._base_schedule.items():
            complete_schedule[day] = {}
            for class_time, class_info in day_schedule.items():
                updated_info = class_info.copy()
                updated_info['speakers'] = class_info['speakers'] | {self.teacher_name}
                updated_info['teacher'] = self.teacher_name
                complete_schedule[day][class_time] = updated_info
        return complete_schedule

    def get_class_info(self, recording_time: datetime) -> Optional[Dict]:
        """Get class information based on recording time"""
        day = recording_time.strftime('%A')
        if day not in self.schedule:
            return None
            
        current_time = time(recording_time.hour, recording_time.minute)
        
        # Find the closest class time that's within 30 minutes of the recording
        day_schedule = self.schedule[day]
        for class_time, class_info in day_schedule.items():
            time_diff = abs(
                (current_time.hour * 60 + current_time.minute) - 
                (class_time.hour * 60 + class_time.minute)
            )
            if time_diff <= 30:  # Within 30 minutes of class time
                return class_info
        return None
    
    def get_all_speakers(self) -> Set[str]:
        """Get a set of all speakers across all classes"""
        speakers = {self.teacher_name}  # Always include teacher
        for day_schedule in self._base_schedule.values():
            for class_info in day_schedule.values():
                speakers.update(class_info['speakers'])
        return speakers