import matplotlib
matplotlib.use('Agg')

from .managers import WhisperManager, MediapipeManager, OpenSmileManager
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
from .modules.gnn_modules.build_graph import *
from .modules.gnn_modules.graphconv import SAGEConv,HeteroGraphConv
from .modules.gnn_modules.self_att import Attention
from .modules.mult_modules.mulT import MULTModel
from .misc import chunk_dataset
import whisper_timestamped as whisper
import yaml
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import dgl
import configparser
from moviepy.editor import VideoFileClip
from typing import Union
import logging


import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


class RAPIDModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.seed = self.config['random_seed']
        self.device = self.config['device']
        if self.device != 'cpu':
            if torch.cuda.is_available() and isinstance(self.device, int):
                self.device = f'cuda:{self.device}'
            else:
                raise ValueError('CUDA is not available or device is not an integer')
        
        self.chunk_size = self.config['chunk_size']
        
        self.num_labels = self.config['num_labels']
        self.dropout = nn.Dropout(self.config['dropout'])
        self.loss_type = self.config['loss']
        ## col
        self.y_col = self.config['y_col']
        self.modal = self.config['modal']
        self.t, self.a, self.v  = self.modal.split('_') 
        
        self.att = self.config['att']
        self.tonly,self.aonly,self.vonly= self.att.split('_')
        self.output_dim = self.config['num_labels']
        self.txt_col = 'asr_body_pre'
        self.token_num = self.config['num_token']

        pretrained = 'roberta-base'
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained, do_lower_case=True)
        self.model = RobertaModel.from_pretrained(pretrained)
        self.keywords = json.load(open(self.config['keyword_path']))[:self.token_num]
        self._update_keywords()


        
        self.a_hidden = int(self.config['hidden'][self.a])
        self.t_hidden = int(self.config['hidden'][self.t])
        self.v_hidden = int(self.config['hidden'][self.v])
        
        ## dgl
        # Inter Relation define
        self.gnn_size = 768
        self.agg_type = self.config['agg_type']
        self.hetero_type = self.config['hetero_type']
        if self.config['rel_type'] == 'v':
            rel_names = {'vk': (int(self.v_hidden)*3, int(self.t_hidden)),
                         'kv': (int(self.t_hidden), int(self.v_hidden)*3)}
        elif self.config['rel_type'] == 'a':
            rel_names = {'ak': (int(self.a_hidden)*3, int(self.t_hidden)),
                         'ka': (int(self.t_hidden), int(self.a_hidden)*3),
                         'vk': (int(self.v_hidden)*3, int(self.t_hidden)),
                         'kv': (int(self.t_hidden), int(self.v_hidden)*3)}
        elif self.config['rel_type'] == 'va':
            rel_names = {'ak': (int(self.a_hidden)*3, int(self.t_hidden)),
                         'ka': (int(self.t_hidden), int(self.a_hidden)*3),
                         'vk': (int(self.v_hidden)*3, int(self.t_hidden)),
                         'kv': (int(self.t_hidden), int(self.v_hidden)*3)}
        
        # Model init
        mod_dict = {rel : SAGEConv((src_dim, dst_dim), self.gnn_size,
                           aggregator_type = self.agg_type)\
                          for rel,(src_dim, dst_dim) in rel_names.items()}

        self.conv = HeteroGraphConv(mod_dict, aggregate=self.hetero_type)
        
        self.v_lstm = nn.LSTM(input_size=self.v_hidden,
                             hidden_size=self.v_hidden,
                             num_layers=2,
                             bidirectional=True,
                             batch_first=True)
        
        self.a_lstm = nn.LSTM(input_size=self.a_hidden,
                             hidden_size=self.a_hidden,
                             num_layers=2,
                             bidirectional=True,
                             batch_first=True)
        
        self.v_atten = Attention(self.device,int(self.v_hidden *2), batch_first=True)  # 2 is bidrectional
        self.a_atten = Attention(self.device,int(self.a_hidden *2), batch_first=True)  # 2 is bidrectional
        self.dropout = nn.Dropout(self.config['dropout'])

        # model & hidden
        if self.config['graphuse']:
            self.MULTModel = MULTModel(self.config, use_origin=True)
        else:
            self.MULTModel = MULTModel(self.config, use_origin=False)
            
        self.fc1 = nn.Linear(self.t_hidden, int(self.t_hidden/2))
        self.fc2 = nn.Linear(int(self.t_hidden/2), self.output_dim)

        self.whisper_manager = None
        self.frame_manager = None
        self.gesture_manager = None
        self.audio_manager = None

        from captum.attr import IntegratedGradients
        self.ig = IntegratedGradients(self.forward_global_wrapper)

    def _update_keywords(self):
        self.tokenizer.add_tokens(self.keywords, special_tokens=False)
        self.model.resize_token_embeddings(len(self.tokenizer))
        keyword_tokens = [self.tokenizer.encode(keyword, add_special_tokens=False)[0] for keyword in self.keywords]
        self.keyword_token = torch.tensor(keyword_tokens)
        self.key_embed = self.model(input_ids=self.keyword_token.unsqueeze(0))[0][0]


    # @torch.inference_mode()
    def forward_inference(self, txt, aud, vid, **kwargs):
        """
        txt: (1, seq_len)
        aud: (1, seq_len, audio_feature_dim)
        vid: (1, seq_len, video_feature_dim)
        """
        # 0. Input
        num_chunk, seq_ = txt.shape
        v_h,_ = self.v_lstm(vid)
        v_h, v_att_score = self.v_atten(v_h)
        
        a_h,_ = self.a_lstm(aud)
        a_h, a_att_score = self.a_atten(a_h)
        

        def historic_feat(feat):
            next_ = torch.cat([feat[1:,:,:], feat[0,:,:].unsqueeze(0)],axis=0)
            past_ = torch.cat([feat[-1,:,:].unsqueeze(0),feat[:-1,:,:]],axis=0)

            feat =torch.cat([feat,next_,past_],axis=2)
            return feat
        
        vid_h = historic_feat(v_h)
        aud_h = historic_feat(a_h)
        
        
        # 1. Speech-gesture graph encoder
        if self.config['graphuse']:
            bg = dgl.batch(self.g_list)#.to(device)    
            bg.ndata['features']['a'] = a_h.reshape(-1,int(self.a_hidden*2))
            bg.ndata['features']['v'] = v_h.reshape(-1,int(self.v_hidden*2))
            bg.ndata['features']['k'] = self.key_embed.repeat(num_chunk,1,1).reshape(-1,self.t_hidden)
            
            if self.config['edge_weight']:
                mod_args = bg.edata['weights'] #{'edge_weight': bg.edata['weights']}
            else:
                mod_args = None
            gnn_h = self.conv(g=bg, inputs=bg.ndata['features'], mod_args=mod_args)
        
        
        # 2. Gesture-aware embedding update
        if self.config['update']:
            key_embed = gnn_h['k'].reshape(num_chunk,-1,self.gnn_size) # 32 x 30 x 20

            with torch.no_grad():
                new_embedding = self.model.embeddings.word_embeddings.weight.data.clone()
                new_embedding[self.keyword_token] = key_embed[0] 
                self.model.embeddings.word_embeddings.weight.set_(new_embedding)

        if self.config['graphuse']:
            if self.config['rel_type'] == 'v':
                aud_h = gnn_h['v'].reshape(num_chunk,-1,self.gnn_size) # 32 x 30 x 20 [batch*node, hidden_dim]
                vid_h = aud_h
            elif self.config['rel_type'] == 'a':
                vid_h = gnn_h['a'].reshape(num_chunk,-1,self.gnn_size) # 32 x 30 x 20
                aud_h = vid_h 
                
            elif self.config['rel_type'] == 'va':
                aud_h = gnn_h['a'].reshape(num_chunk,-1,self.gnn_size) # 32 x 30 x 20 [batch*node, hidden_dim]
                vid_h = gnn_h['v'].reshape(num_chunk,-1,self.gnn_size) # 32 x 30 x 20
        txt_h = self.model(input_ids =txt)
        
        # 3. Multimodal Fusion Encoder
        with torch.cuda.amp.autocast():
            relation_h, _, att_ls = self.MULTModel(txt_h[0], aud_h, vid_h)
        last_h_l = txt_h[1] + relation_h
        
        # 5 Integrated Gradients
        baseline = torch.zeros_like(last_h_l).to(self.device)
        logits = self.forward_global_wrapper(last_h_l)  # shape: [1, num_labels]
        target_class = logits[0].argmax()  # batch dimension 제거
        
        attributions = self.ig.attribute( # shape: [batch_size, feature_dim]
            last_h_l,
            baseline,
            target=target_class,
            n_steps=50
        )
        
        chunk_scores = torch.norm(attributions, p=2, dim=1) # shape: [batch_size]
        chunk_scores = F.softmax(chunk_scores, dim=0) # shape: [batch_size]
        
        # 6. Token-level importance scores
        token_scores = []
        max_att_values = []

        for i in range(num_chunk):
            chunk_att = att_ls[i]
            token_importance = chunk_att.mean(dim=1)
            max_att_values.append(token_importance.max().item())

        for i in range(num_chunk):
            chunk_att = att_ls[i] # shape: [chunk_size, chunk_size]
            token_importance = chunk_att.mean(dim=1) 
            
            token_importance = (token_importance - token_importance.min()) / \
                            (token_importance.max() - token_importance.min())
            
            weighted_importance = token_importance * chunk_scores[i]
            token_scores.append(weighted_importance)

        all_token_scores = F.softmax(torch.cat(token_scores), dim=0) * 100

        return logits[0], all_token_scores

    def forward_global_wrapper(self, all_hidden_states):
        chunk_logits = []
        for hidden in all_hidden_states:
            intermediate = F.relu(self.fc1(hidden))
            logits = self.fc2(intermediate)
            chunk_logits.append(logits)
            
        mean_logits = torch.stack(chunk_logits).mean(dim=0)
        return mean_logits.unsqueeze(0)


    def detect_highlights(self, all_token_scores, window_size=10):
        best_peak = 0
        best_importance_sum = 0
        best_start = 0
        best_end = 0
        
        num_tokens = len(all_token_scores)

        # Rest of the tokens are truncated
        self.all_tokens_with_timestamp = self.all_tokens_with_timestamp.iloc[:num_tokens]
        self.all_tokens_with_timestamp['importance'] = [score.item() for score in all_token_scores]

        # Find best window
        for i in range(num_tokens - window_size + 1):
            peak = all_token_scores[i:i+window_size].max().item()
            importance_sum = all_token_scores[i:i+window_size].sum()

            # Update best peak
            if peak > best_peak:
                best_peak = peak
                best_importance_sum = importance_sum
                best_start = self.all_tokens_with_timestamp.iloc[i]['start']
                best_end = self.all_tokens_with_timestamp.iloc[i+window_size-1]['end']
            elif peak == best_peak:
                # Update best importance sum
                if importance_sum > best_importance_sum:
                    best_peak = peak
                    best_importance_sum = importance_sum
                    best_start = self.all_tokens_with_timestamp.iloc[i]['start']
                    best_end = self.all_tokens_with_timestamp.iloc[i+window_size-1]['end']

        return (best_start, best_end)


    def extract_features(self, video_path):
        features = {}

        # 1. Transcription
        audio_path = extract_audio(video_path)
        audio = whisper.load_audio(audio_path)
                
        with WhisperManager(model_name="turbo", device=self.device) as whisper_model:
            transcription = whisper.transcribe(
                model=whisper_model,
                audio=audio,
                language="en",
                detect_disfluencies=True,
                condition_on_previous_text=False
            )
        print('Transcription done')
        
        # 2. Frame Extraction
        cap = cv2.VideoCapture(video_path)
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        hop = round(fps / 1)
        frames = []
        curr_frame = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if curr_frame % hop == 0:
                frames.append({
                    'time': int(curr_frame // hop)+1,
                    'frame': frame
                })
            curr_frame += 1
        
        cap.release()
                
        frame_times = [frame['time'] for frame in frames]
        print('Frames extracted')
        
        # 3. Frame Alignment & Gesture Analysis
        text_bind = pd.DataFrame()
        pose_bind = pd.DataFrame()
        with MediapipeManager() as holistic:
            for segment in transcription["segments"]:
                for word_info in segment["words"]:
                    start_time = word_info["start"]
                    end_time = word_info["end"]
                    text = word_info["text"]
                    
                    closest_index = int(np.round(start_time)) if np.round(start_time) <= end_time else int(start_time)
                    try:
                        frame = frames[frame_times.index(closest_index)]["frame"]
                    except:
                        continue
                        
                    results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    pose_data = self._extract_pose_data(results, holistic)
                    
                    text_bind = pd.concat([text_bind, pd.DataFrame({
                        'token': text,
                        'start': start_time,
                        'end': end_time
                    }, index=[0])], ignore_index=True)
                    
                    pose_bind = pd.concat([pose_bind, pd.json_normalize(pose_data)], ignore_index=True)
        
        print('Frames aligned and pose extracted')
        
        # 4. Audio Feature Extraction
        with OpenSmileManager() as smile:
            zero_point = text_bind.iloc[0]['start']
            opensmile_bind = pd.DataFrame()
            
            for _, token in text_bind.iterrows():
                features = smile.process_file(
                    audio_path,
                    start=token['start']-zero_point,
                    end=token['end']-zero_point
                )
                opensmile_bind = pd.concat([opensmile_bind, features], axis=0)

        print('Opensmile features extracted')

        text_bind.reset_index(drop=True, inplace=True)
        pose_bind.reset_index(drop=True, inplace=True)
        opensmile_bind.reset_index(drop=True, inplace=True)
        
        # Prepare final binds
        pose_bind = pose_bind.interpolate()
        audio_bind = opensmile_bind.interpolate()

        self.all_tokens_with_timestamp = text_bind

        return text_bind, audio_bind, pose_bind


    def _extract_pose_data(self, results, holistic):
        """Helper method to extract pose data from mediapipe results"""
        data = {}
        if results.pose_landmarks:
            for mark, data_point in zip(holistic.PoseLandmark, results.pose_landmarks.landmark):
                data[f'{mark.name}_x'] = data_point.x
                data[f'{mark.name}_y'] = data_point.y
                data[f'{mark.name}_z'] = data_point.z
        else:
            for mark in holistic.PoseLandmark:
                data[f'{mark.name}_x'] = np.nan
                data[f'{mark.name}_y'] = np.nan
                data[f'{mark.name}_z'] = np.nan
        return data

    def predict(self, video_path):
        # Extract all features using context managers
        text_bind, audio_bind, pose_bind = self.extract_features(video_path)
        
        # Process features
        audio_feats, video_feats, audio_feats_with_tokens, video_feats_with_tokens, adj_matrices, text_bind = chunk_dataset(text_bind, 
                                                                                                                            audio_bind, 
                                                                                                                            pose_bind, 
                                                                                                                            chunk_size=self.chunk_size, 
                                                                                                                            user='john', 
                                                                                                                            sex='male',
                                                                                                                            disfluency_tokens=self.config['keyword_path'])
        
        text_bind = pd.DataFrame(text_bind)
        # Tokenize text
        text_bind[self.txt_col] = text_bind[self.txt_col].map(
            lambda x: self.tokenizer.encode(
                str(x),
                padding='max_length',
                max_length=self.chunk_size,
                truncation=True,
            )
        )

        # Build graph if needed
        if self.config['graphuse']:
            self.g_list = build_graph(
                self.config,
                audio_feats,
                video_feats,
                adj_matrices
            ).data_load(self.device)
        
        # Handle NaN values
        audio_feats_with_tokens = np.nan_to_num(audio_feats_with_tokens).astype('float64')
        video_feats_with_tokens = np.nan_to_num(video_feats_with_tokens).astype('float64')

        txt = torch.tensor(text_bind[self.txt_col], dtype=torch.long).to(self.device)
        aud = torch.tensor(audio_feats_with_tokens, dtype=torch.float32).to(self.device)
        vid = torch.tensor(video_feats_with_tokens, dtype=torch.float32).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits, token_scores = self.forward_inference(txt, aud, vid)

        pred = logits.argmax().item()
        highlight_timestamp = self.detect_highlights(token_scores)
    
        return logits, pred, highlight_timestamp, self.all_tokens_with_timestamp
        


def extract_audio(
    video_path: Union[str, Path],
    sample_rate: int = 16000,
    audio_channels: int = 1,
    format: str = 'wav'
) -> str:
    try:
        # 1. 경로 처리
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
            
        audio_path = video_path.with_suffix(f'.{format}')
        
        # 2. 출력 디렉토리가 없다면 생성
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 3. 비디오 로딩과 오디오 추출
        try:
            video = VideoFileClip(str(video_path))
        except Exception as e:
            raise ValueError(f"비디오 파일 로딩 실패: {str(e)}")
            
        if video.audio is None:
            raise ValueError(f"비디오에 오디오 트랙이 없습니다: {video_path}")
        
        try:
            # 오디오 속성 설정
            audio = video.audio
            audio.fps = sample_rate  # 샘플레이트 설정
            
            # 오디오 추출 및 저장
            audio.write_audiofile(
                str(audio_path),
                fps=sample_rate,
                nbytes=2,  # 16-bit audio
                codec='pcm_s16le' if format == 'wav' else None,
                ffmpeg_params=[
                    '-ac', str(audio_channels),
                    '-ar', str(sample_rate)
                ] if format == 'wav' else None,
                logger=None  # moviepy의 진행률 표시 비활성화
            )
        except Exception as e:
            raise ValueError(f"오디오 추출 중 오류 발생: {str(e)}")
        finally:
            # 4. 리소스 정리
            video.close()
            
        # 5. 출력 파일 생성 확인
        if not audio_path.exists():
            raise RuntimeError(f"오디오 파일이 생성되지 않았습니다: {audio_path}")
            
        if audio_path.stat().st_size == 0:
            audio_path.unlink()  # 빈 파일 삭제
            raise RuntimeError(f"생성된 오디오 파일이 비어있습니다: {audio_path}")
            
        return str(audio_path)
        
    except FileNotFoundError as e:
        logging.error(f"파일 없음 에러: {str(e)}")
        raise
    except ValueError as e:
        logging.error(f"처리 에러: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"예상치 못한 에러: {str(e)}")
        raise


# For debugging
if __name__ == '__main__':
    config = configparser.ConfigParser()
    with open('/workspace/MMATD/rapid-api-server/config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print('Starting...')
    model = RAPIDModel(config)
    print('Model loaded...')
    pred, att_vl, v_att_score = model.predict('/workspace/dataset/final_video.mp4')
    print(f'Prediction: {pred}')
    print('Prediction done...')