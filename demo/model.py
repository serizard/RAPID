import matplotlib
matplotlib.use('Agg')

from process_input import WhisperManager, MediapipeManager, OpenSmileManager, chunk_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
from modules.gnn_modules.build_graph import *
from modules.gnn_modules.graphconv import SAGEConv,HeteroGraphConv
from modules.gnn_modules.self_att import Attention
from modules.mult_modules.mulT import MULTModel
import whisper_timestamped as whisper
import yaml
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import dgl
from misc import add_special_tokens_to_features, create_feature_adjacency_matrices, audio_col, video_col
from moviepy.editor import VideoFileClip
from typing import Union
import logging




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
        
        self._load_model()


    def _update_keywords(self):
        self.tokenizer.add_tokens(self.keywords, special_tokens=False)
        self.model.resize_token_embeddings(len(self.tokenizer))
        keyword_tokens = [self.tokenizer.encode(keyword, add_special_tokens=False)[0] for keyword in self.keywords]
        self.keyword_token = torch.tensor(keyword_tokens)
        self.key_embed = self.model(input_ids=self.keyword_token.unsqueeze(0))[0][0]


    @torch.inference_mode()
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
            relation_h,_, att_ls =self.MULTModel(txt_h[0], aud_h, vid_h) # 32 x 20
        last_h_l = txt_h[1]+relation_h
        
        # 4. Aphasia Type Detection
        logits = self.fc2(F.relu(self.fc1(last_h_l)))

        return logits,att_ls,v_att_score

    def _load_model(self):
        state_dict = torch.load(self.config['checkpoint_path'], map_location='cpu')['state_dict']
        self.load_state_dict(state_dict, strict=False)
        self.eval()
    
    
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

        print(len(text_bind), len(audio_bind), len(pose_bind))
        text_bind.to_csv('D:/aphasia/MMATD/src/temp/text_bind.csv')


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
                                                                                                                           disfluency_tokens='D:/aphasia/MMATD/dataset/_disfluency_tk_300.json')
        
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
        # logits: (num_chunk, num_labels)
        # att_ls: (num_chunk, chunk_size, chunk_size)d
        # v_att_score: (num_chunk, chunk_size+2)
        logits, att_ls, v_att_score = self.forward_inference(txt, aud, vid)
    
        # Calculate prediction
        mean_logits = logits.mean(0)
        pred = mean_logits.argmax().item()

        return pred, att_ls, v_att_score
        

def extract_audio(
    video_path: Union[str, Path],
    sample_rate: int = 16000,
    audio_channels: int = 1,
    format: str = 'wav'
) -> str:
    try:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"No video file found: {video_path}")
            
        audio_path = video_path.with_suffix(f'.{format}')
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            video = VideoFileClip(str(video_path))
        except Exception as e:
            raise ValueError(f"Failed to load a video: {str(e)}")
            
        if video.audio is None:
            raise ValueError(f"No audio track in a video: {video_path}")
        
        try:
            audio = video.audio
            audio.fps = sample_rate 
            
            audio.write_audiofile(
                str(audio_path),
                fps=sample_rate,
                nbytes=2,  # 16-bit audio
                codec='pcm_s16le' if format == 'wav' else None,
                ffmpeg_params=[
                    '-ac', str(audio_channels),
                    '-ar', str(sample_rate)
                ] if format == 'wav' else None,
                logger=None 
            )
        except Exception as e:
            raise ValueError(f"Error occurred during audio extraction: {str(e)}")
        finally:
            video.close()
            
        if not audio_path.exists():
            raise RuntimeError(f"No audio file generated: {audio_path}")
            
        if audio_path.stat().st_size == 0:
            audio_path.unlink()
            raise RuntimeError(f"Empty audio file: {audio_path}")
            
        return str(audio_path)
        
    except FileNotFoundError as e:
        logging.error(f"No files: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected Error: {str(e)}")
        raise
