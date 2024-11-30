import base64
from pathlib import Path

def get_image_base64(image_path: Path) -> str:
    """Convert image to base64 encoding."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        return ""