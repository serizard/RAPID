import tempfile
from pathlib import Path
import shutil
from moviepy.editor import VideoFileClip
import logging

def save_upload_file_temp(upload_file) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(upload_file.file, temp_file)
            return Path(temp_file.name)
    except Exception as e:
        logging.error(f"Error saving uploaded file: {e}")
        raise

def extract_audio(video_path: Path) -> Path:
    try:
        audio_path = video_path.with_suffix('.wav')
        video = VideoFileClip(str(video_path))
        video.audio.write_audiofile(str(audio_path))
        return audio_path
    except Exception as e:
        logging.error(f"Error extracting audio: {e}")
        raise