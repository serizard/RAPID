import cv2
import whisper_timestamped as whisper
import subprocess
import numpy as np
import mediapipe as mp
import opensmile
import pandas as pd
import multiprocessing
from misc import add_special_tokens_to_features, create_feature_adjacency_matrices, audio_col, video_col
import os
import json
import torch


import whisper_timestamped as whisper
import subprocess
import os
import torch
import numpy as np
from pathlib import Path
import soundfile as sf
import pandas as pd
import cv2
import mediapipe as mp


# class Transcriptor:
#     def __init__(self, model_size: str = "tiny", device: str = "cpu"):
#         """
#         Initialize the speech processor with specified model size and device.
        
#         Args:
#             model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
#             device: Computing device ("cuda" or "cpu")
#         """
#         self.device = device
#         try:
#             self.model = whisper.load_model(name='turbo', device=self.device)
#         except Exception as e:
#             print(f"Error loading whisper model: {e}")
#             raise
#         self.model_size = model_size

#     def _extract_audio(self, video_path):
#         """
#         Extract audio from video file with robust error handling
#         """
#         try:
#             video_path = Path(video_path)
#             # Use wav instead of mp3 for better compatibility
#             output_path = video_path.with_suffix('.wav')
            
#             if not video_path.exists():
#                 raise FileNotFoundError(f"Video file not found: {video_path}")
            
#             # Use more stable ffmpeg settings
#             command = [
#                 'ffmpeg',
#                 '-i', str(video_path),
#                 '-vn',                    # No video
#                 '-acodec', 'pcm_s16le',  # PCM format
#                 '-ar', '16000',          # 16kHz sample rate
#                 '-ac', '1',              # Mono
#                 '-loglevel', 'warning',
#                 '-y',                    # Overwrite output
#                 str(output_path)
#             ]
            
#             # Run ffmpeg with shell=False for better stability
#             result = subprocess.run(
#                 command,
#                 shell=False,
#                 check=True,
#                 capture_output=True,
#                 text=True
#             )
            
#             if not output_path.exists():
#                 raise RuntimeError(f"Failed to create audio file. ffmpeg output: {result.stderr}")
                
#             return str(output_path)
            
#         except subprocess.CalledProcessError as e:
#             print(f"FFmpeg error: {e.stderr}")
#             raise RuntimeError(f"Failed to extract audio: {e}")
#         except Exception as e:
#             print(f"Error extracting audio: {e}")
#             raise

#     def _load_audio(self, audio_path):
#         """
#         Load audio file using soundfile instead of whisper's load_audio
#         """
#         try:
#             audio, sr = sf.read(audio_path)
#             # Convert to float32 and normalize
#             audio = audio.astype(np.float32) / np.max(np.abs(audio))
#             return audio
#         except Exception as e:
#             raise RuntimeError(f"Failed to load audio file: {e}")

#     def _add_timestamps(self, result):
#         """
#         Convert basic whisper output to timestamped format
#         """
#         segments = []
        
#         for segment in result["segments"]:
#             words = []
#             text_parts = segment["text"].split()
#             segment_duration = segment["end"] - segment["start"]
#             word_duration = segment_duration / len(text_parts)
            
#             current_time = segment["start"]
#             for word in text_parts:
#                 word_info = {
#                     "text": word,
#                     "start": current_time,
#                     "end": current_time + word_duration
#                 }
#                 words.append(word_info)
#                 current_time += word_duration
            
#             segment["words"] = words
        
#         return result

#     def transcribe(self, video_path: str):
#         """
#         Transcribe audio from video with robust error handling
#         """
#         try:
#             # Extract audio
#             audio_path = self._extract_audio(video_path)
            
#             # Load audio file
#             try:
#                 audio = self._load_audio(audio_path)
#             except Exception as e:
#                 raise RuntimeError(f"Failed to load audio file: {e}")

#             # Transcribe
#             try:
#                 with torch.inference_mode():
#                     result = self.model.transcribe(
#                         audio,
#                         language='en',
#                         task='transcribe',
#                         fp16=False  # Use fp32 for better stability
#                     )
#                     # Add word-level timestamps
#                     result = self._add_timestamps(result)
#             except Exception as e:
#                 raise RuntimeError(f"Transcription failed: {e}")

#             return result, audio_path

#         except Exception as e:
#             print(f"Transcription error: {e}")
#             # Return empty transcription and audio path if transcription fails
#             empty_transcription = {
#                 "segments": [],
#                 "text": ""
#             }
#             return empty_transcription, None
#         finally:
#             # Clean up the temporary file in finally block
#             try:
#                 if audio_path and os.path.exists(audio_path):
#                     os.remove(audio_path)
#             except Exception as e:
#                 print(f"Cleanup error: {e}")


class Transcriptor:
    def __init__(self, model_size: str = "turbo ", device: str = "cuda"):
        """
        Initialize the speech processor with specified model size and device.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large", "turbo")
            device: Computing device ("cuda" or "cpu")
        """

        self.device='cpu'
        self.model = whisper.load_model(name=model_size, device=self.device)
        self.model_size = model_size


        # self.device = device
        # print(self.device)
        # print(type(self.device))
        # self.model = whisper.load_model(model_size, device=self.device)
        # self.model_size = model_size

    @staticmethod
    def _extract_audio(video_path):
        output_path = video_path.replace(".mp4", ".mp3")
        command = f'ffmpeg -i "{video_path}" -vn -ar 16000 -ac 1 -ab 192k -loglevel panic -f mp3 -y "{output_path}"'
        subprocess.run(command, shell=True, check=True)        
        return output_path

    def transcribe(self, video_path: str):
        audio_path = self._extract_audio(video_path)

        audio = whisper.load_audio(file=audio_path, sr=16000)

        with torch.inference_mode():
            transcription = whisper.transcribe(
                model=self.model,
                audio=audio,
                language='en',
                detect_disfluencies=True,
                condition_on_previous_text=False,
            )

        return transcription, audio_path

class GestureExtractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
    
    def extract_pose_bind(self, text_bind):
        pose_bind = pd.DataFrame()

        with self.mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as holistic:
            for token_img_path in text_bind['token_img_path'].tolist():
                image = cv2.imread(token_img_path)
                results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                                           
                dic = {'token_img_path': token_img_path}
                if results.pose_landmarks:
                    for mark, data_point in zip(self.mp_holistic.PoseLandmark, results.pose_landmarks.landmark):
                        dic[f'{mark.name}_x'] = data_point.x
                        dic[f'{mark.name}_y'] = data_point.y
                        dic[f'{mark.name}_z'] = data_point.z
                else:
                    for mark in self.mp_holistic.PoseLandmark:
                        dic[f'{mark.name}_x'] = np.nan
                        dic[f'{mark.name}_y'] = np.nan
                        dic[f'{mark.name}_z'] = np.nan
                        
                pose_bind = pd.concat([pose_bind, pd.json_normalize(dic)], ignore_index=True)
            
        pose_bind = pose_bind[['token_img_path'] + video_col]
        return pose_bind[video_col].interpolate()

class AudioExtractor:
    def __init__(self):
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
            num_workers=multiprocessing.cpu_count()*2
        )
    
    def extract_opensmile_bind(self, text_bind, audio_path):
        zero_point = text_bind.iloc[0]['start']
        opensmile_bind = pd.DataFrame()
        for i, token in text_bind.iterrows():
            y = self.smile.process_file(audio_path, start=token['start']-zero_point, end=token['end']-zero_point)
            opensmile_bind = pd.concat([opensmile_bind, y], axis=0)

        return opensmile_bind[audio_col].interpolate()


class FrameExtractor:
    def __init__(self):
        self.frames = None
        self.frame_times = None
    
    def clear(self):
        self.frames = None
        self.frame_times = None
        
    def extract_frames(self, video_path, fps=1):
        cap = cv2.VideoCapture(video_path)
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        hop = round(fps / 1)

        self.frames = []
        curr_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if curr_frame % hop == 0:
                self.frames.append({'time': int(curr_frame // hop)+1, 'frame': frame})
            curr_frame += 1

        cap.release()

        self.frame_times = [frame['time'] for frame in self.frames]

    def align_frames(self, transcription):
        if not self.frames:
            raise ValueError("Frames not extracted yet")
    
        text_bind = pd.DataFrame()
        id = 0
        for segment in transcription["segments"]:
            for word_info in segment["words"]:
                start_time, end_time, text = word_info["start"], word_info["end"], word_info["text"]
                closest_index = int(np.round(start_time)) if np.round(start_time) <= end_time else int(start_time)
                try:
                    frame = self.frames[self.frame_times.index(closest_index)]["frame"]
                except:
                    continue
                gesture_fp = os.path.join('./temp/gestures', f"{id}_gesture.jpg")
                cv2.imwrite(gesture_fp, frame)

                text_bind.append({
                    "token": text,
                    "start": start_time,
                    "end": end_time,
                    "token_img_path": gesture_fp
                })
                id += 1
        
        return pd.DataFrame(text_bind)
    

def chunk_dataset(text_bind, opensmile_bind, pose_bind, chunk_size=50, **kwargs):
    combined_binds = pd.concat([text_bind, pose_bind, opensmile_bind], axis=1)

    user_name = kwargs.get('user', None)
    sex = kwargs.get('sex', None)

    dataset_chunks = []
    chunk_infos = {}
    for i in range(0, len(combined_binds), chunk_size):
        if i + chunk_size <= len(combined_binds):
            dataset_chunks.append(combined_binds.iloc[i:i+chunk_size])
            chunk_infos['user_name'][i] = user_name
            chunk_infos['sex'][i] = sex
            chunk_infos['asr_body_pre'] = ' '.join(combined_binds.iloc[i:i+chunk_size]['token'])
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

    return audio_feats, video_feats, audio_feats_with_additional_tokens, video_feats_with_additional_tokens, adj_matrices