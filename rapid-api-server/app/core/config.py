import yaml
from pathlib import Path
from dotenv import load_dotenv
import os


def load_config():
    current_dir = Path(__file__).parent 
    root_dir = current_dir.parent.parent 
    config_path = root_dir / "config" / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class Settings:
    def __init__(self):
        self.config = load_config()
        current_dir = Path(__file__).parent
        root_dir = current_dir.parent.parent
        self.MODEL_PATH = root_dir / "checkpoints" / "model.ckpt"
        self.API_KEY = '695f9eb5021752735066a7d14fa166fa5007ff6a2dcaee6b8dae9cb4a4a69b09'
        self.KEYWORD_PATH = '/workspace/MMATD/rapid-api-server/app/models/_disfluency_tk_300.json'