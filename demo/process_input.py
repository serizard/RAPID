import os
import json
import cv2
import numpy as np
import pandas as pd
from misc import add_special_tokens_to_features, create_feature_adjacency_matrices, audio_col, video_col


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


def chunk_dataset(text_bind, opensmile_bind, pose_bind, chunk_size=50, **kwargs):
    combined_binds = pd.concat([text_bind, pose_bind, opensmile_bind], axis=1)

    user_name = kwargs.get('user', None)
    sex = kwargs.get('sex', None)

    dataset_chunks = []
    chunk_infos = {"user_name": {}, "sex": {}, "asr_body_pre": {}}

    print('length of combined binds:', len(combined_binds))
    for i in range(0, len(combined_binds), chunk_size):
        if i + chunk_size <= len(combined_binds):
            dataset_chunks.append(combined_binds.iloc[i:i+chunk_size])
            chunk_infos['user_name'][i] = user_name
            chunk_infos['sex'][i] = sex
            chunk_infos['asr_body_pre'][i] = ' '.join(combined_binds.iloc[i:i+chunk_size]['token'])
        elif i == 0:
            raise ValueError("Chunk size is too large for the dataset")
        else:
            continue
    
    audio_feats_with_additional_tokens = []
    video_feats_with_additional_tokens = []
    audio_feats = []
    video_feats = []

    for chunk in dataset_chunks:
        audio_feat = np.array(chunk[audio_col].values)
        video_feat = np.array(chunk[video_col].values)
        audio_feats.append(audio_feat)
        video_feats.append(video_feat)

        audio_feats_with_additional_tokens.append(add_special_tokens_to_features(audio_feat))
        video_feats_with_additional_tokens.append(add_special_tokens_to_features(video_feat))

    audio_feats = np.stack(audio_feats)
    video_feats = np.stack(video_feats)
    audio_feats_with_additional_tokens = np.stack(audio_feats_with_additional_tokens)
    video_feats_with_additional_tokens = np.stack(video_feats_with_additional_tokens)

    disfluency_tokens = json.load(open(kwargs.get('disfluency_tokens', None), 'r'))

    adj_matrices = create_feature_adjacency_matrices(
        dataset_chunks, 
        disfluency_tokens, 
        video_col, 
        audio_col,
        std_multiplier=1.5
    )

    return audio_feats, video_feats, audio_feats_with_additional_tokens, video_feats_with_additional_tokens, adj_matrices, chunk_infos
