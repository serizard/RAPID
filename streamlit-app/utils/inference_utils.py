import requests
from pathlib import Path
from config import Config

def get_inference_result(video_path, url="http://localhost:8000/predict"):
    video_path = Path(video_path)

    headers = {
        "X-API-Key": Config.INFERENCE_API_KEY
    }
    
    files = {
        "file": (video_path.name, open(video_path, "rb"), "video/mp4")
    }
    
    try:
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()
        
        result = response.json()
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"API 호출 중 오류 발생: {e}")
        raise
    finally:
        files["file"][1].close() 