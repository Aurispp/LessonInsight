# LessonInsight

A Python application for recording audio and transcribing it using Whisper.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/lessoninsight.git
cd lessoninsight
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## System Requirements

- Python 3.8 or higher
- FFmpeg
- PortAudio

### MacOS
```bash
brew install ffmpeg portaudio
```

### Windows
Download FFmpeg from the official website and add it to your system PATH.

## Usage

[Add usage instructions here]

## Project Structure
```
lessoninsight/
├── src/
│   ├── audio/
│   │   ├── recorder.py      # Audio recording functionality
│   │   └── transcriber.py   # Whisper transcription functionality
│   └── utils/
│       └── file_handler.py  # File management utilities
└── audio_files/            # Directory for audio recordings (not tracked in git)
```

## License

MIT License