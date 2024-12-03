class WhisperManager:
    def __init__(self, model_name="turbo", device="cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        
    def __enter__(self):
        import whisper_timestamped as whisper
        self.model = whisper.load_model(self.model_name, device=self.device)
        return self.model
        
    def __exit__(self, *args):
        if self.model:
            del self.model
            self.model = None
            
class MediapipeManager:
    def __enter__(self):
        import mediapipe as mp
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=True,
            min_detection_confidence=0.5,
            model_complexity=2
        )
        # Store PoseLandmark directly from mp_holistic
        self.PoseLandmark = self.mp_holistic.PoseLandmark
        return self
        
    def __exit__(self, *args):
        """Release resources on context end"""
        if hasattr(self, 'holistic'):
            self.holistic.close()
            
    def process(self, image):
        """Process image through holistic model"""
        return self.holistic.process(image)

class OpenSmileManager:
    def __enter__(self):
        """컨텍스트 시작시 OpenSmile 초기화"""
        import opensmile
        import multiprocessing
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
            num_workers=multiprocessing.cpu_count()
        )
        return self.smile
        
    def __exit__(self, *args):
        """컨텍스트 종료시 리소스 해제"""
        if hasattr(self, 'smile'):
            del self.smile