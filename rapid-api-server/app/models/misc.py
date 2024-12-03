import numpy as np
import pandas as pd
from tqdm import tqdm
import json


def get_labels(aphasia_type):   
   # 소문자로 변환하여 처리
   aphasia_type = aphasia_type.lower()
   
   # 기본값 설정
   type = ""
   ct_label = 0
   wab_label = 0 
   type_label = 0
   flu_label = 0
   com_label = 0
   
   if aphasia_type in ['control', 'notaphasicbywab']:
       type = "Control"
       status = 'Control'
       ct_label = 0
       wab_label = 0
       type_label = 0
       flu_label = 0
       com_label = 0
       
   elif aphasia_type in ['anomic', 'conduction']:
       type = aphasia_type.capitalize()
       status = 'Aphasia'
       ct_label = 1
       wab_label = 1  # Fluent & Comprehends
       type_label = 1  # Fluent
       flu_label = 1  # Fluent
       com_label = 0  # Comprehends
       
   elif aphasia_type in ['wernicke', 'transsensory']:
       type = "Wernicke" if aphasia_type == 'wernicke' else "Trans. Sensory"
       status = 'Aphasia'
       ct_label = 1
       wab_label = 2  # Fluent & Not Comprehends
       type_label = 2  # Non-Comprehension
       flu_label = 1  # Fluent
       com_label = 1  # Not Comprehends
       
   elif aphasia_type in ['broca', 'transmotor']:
       type = "Broca" if aphasia_type == 'broca' else "Trans. Motor"
       status = 'Aphasia'
       ct_label = 1
       wab_label = 3  # Non-Fluent & Comprehends
       type_label = 3  # Non-Fluent
       flu_label = 0  # Non-Fluent
       com_label = 0  # Comprehends
       
   elif aphasia_type in ['global', 'isolation']:
       type = "Global" if aphasia_type == 'global' else "Isolation"
       status = 'Aphasia'
       ct_label = 1
       wab_label = 4  # Non-Fluent & Not Comprehends
       type_label = 3  # Non-Fluent
       flu_label = 0  # Non-Fluent
       com_label = 1  # Not Comprehends
   
   else:
       return None, None, None, None, None, None

   return type, status, ct_label, wab_label, type_label, flu_label, com_label


def add_special_tokens_to_features(features):
    _, feature_dim = features.shape
    
    cls_token = np.ones((1, feature_dim))
    eos_token = -1 * np.ones((1, feature_dim))
    
    augmented_features = np.concatenate([cls_token, features, eos_token], axis=0)
    
    return augmented_features


def calculate_gesture_changes(df, video_col, threshold_multiplier=1.5):
    """프레임별 제스처 변화량 계산"""
    changes = np.zeros(len(df))
    
    for col in video_col:
        if col in df.columns:
            # 각 좌표의 변화량 계산
            diff = np.abs(np.diff(df[col].values, prepend=df[col].values[0]))
            # 임계값 계산
            threshold = np.std(diff) * threshold_multiplier
            # 임계값 이상의 변화가 있는 프레임 표시
            changes += (diff > threshold).astype(float)
    
    # 전체 피처에서 유의미한 변화가 있는 프레임 반환 (numpy array)
    return (changes > 0).astype(bool)

def calculate_audio_changes(df, audio_col, z_score_threshold=1.0):
    """프레임별 오디오 특성 변화량 계산"""
    changes = np.zeros(len(df))
    
    for col in audio_col:
        if col in df.columns:
            values = df[col].values
            if len(values[~np.isnan(values)]) > 0:
                mean = np.nanmean(values)
                std = np.nanstd(values)
                if std > 0:
                    z_scores = np.abs((values - mean) / std)
                    changes += (z_scores > z_score_threshold).astype(float)
    
    return (changes > 0).astype(bool)


def create_feature_adjacency_matrices(dfs, disfluency_tokens, video_col, audio_col):
    """
    Create feature adjacency matrices based on co-occurrence patterns between tokens and multimodal features
    최적화된 numpy 연산 사용
    
    Args:
        dfs: List of DataFrames containing aligned multimodal data
        disfluency_tokens: List of disfluency-related keywords
        video_col: List of gesture feature column names
        audio_col: List of audio feature column names
    
    Returns:
        np.ndarray: Stack of adjacency matrices for each sample
    """
    n_samples = len(dfs)
    n_tokens = len(disfluency_tokens)
    
    # 전체 데이터셋에 대한 통계를 미리 계산
    # Token별 gesture/audio 특성의 평균값을 계산할 arrays
    gesture_values = {token: [] for token in disfluency_tokens}
    audio_values = {token: [] for token in disfluency_tokens}
    
    print("Computing token statistics...")
    for df in tqdm(dfs):
        # 토큰 sequence를 one-hot encoding으로 변환
        token_matrix = np.zeros((len(df), len(disfluency_tokens)))
        for i, token in enumerate(df['token'].values):
            if token.lower() in disfluency_tokens:
                token_idx = disfluency_tokens.index(token.lower())
                token_matrix[i, token_idx] = 1
        
        # Gesture 특성들의 평균 계산
        gesture_features = df[video_col].values
        for j, token in enumerate(disfluency_tokens):
            # 해당 토큰이 있는 프레임의 gesture 특성들
            token_mask = token_matrix[:, j] == 1
            if token_mask.any():
                gesture_values[token].append(gesture_features[token_mask].mean(axis=1))
        
        # Audio 특성들의 평균 계산
        audio_features = df[audio_col].values
        for j, token in enumerate(disfluency_tokens):
            token_mask = token_matrix[:, j] == 1
            if token_mask.any():
                audio_values[token].append(audio_features[token_mask].mean(axis=1))
    
    # Weight 계산
    print("\nComputing weights...")
    gesture_weights = {}
    audio_weights = {}
    
    for token in disfluency_tokens:
        # Gesture weights
        if gesture_values[token]:
            gesture_weights[token] = np.concatenate(gesture_values[token]).mean()
        else:
            gesture_weights[token] = 0.0
            
        # Audio weights    
        if audio_values[token]:
            audio_weights[token] = np.concatenate(audio_values[token]).mean()
        else:
            audio_weights[token] = 0.0
    
    # 인접 행렬 생성 (벡터화된 연산 사용)
    print("\nCreating adjacency matrices...")
    adj_matrices = []
    
    for df in tqdm(dfs):
        # 토큰 시퀀스를 one-hot matrix로 변환
        token_matrix = np.zeros((len(df), n_tokens))
        for i, token in enumerate(df['token'].values):
            if token.lower() in disfluency_tokens:
                token_idx = disfluency_tokens.index(token.lower())
                token_matrix[i, token_idx] = 1
        
        # Weight matrix 생성
        weight_matrix = np.zeros((n_tokens,))
        for j, token in enumerate(disfluency_tokens):
            weight_matrix[j] = gesture_weights[token] + audio_weights[token]
        
        # 최종 인접 행렬 계산 (행렬 곱을 통한 벡터화된 연산)
        adj_matrix = token_matrix * weight_matrix
        adj_matrices.append(adj_matrix)
    
    return np.stack(adj_matrices)


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
        audio_col
    )

    return audio_feats, video_feats, audio_feats_with_additional_tokens, video_feats_with_additional_tokens, adj_matrices, chunk_infos



video_col = ['NOSE_x', 'NOSE_y', 'NOSE_z', 'LEFT_EYE_INNER_x','LEFT_EYE_INNER_y', 'LEFT_EYE_INNER_z', 'LEFT_EYE_x', 'LEFT_EYE_y','LEFT_EYE_z', 
             'LEFT_EYE_OUTER_x', 'LEFT_EYE_OUTER_y','LEFT_EYE_OUTER_z', 'RIGHT_EYE_INNER_x', 'RIGHT_EYE_INNER_y','RIGHT_EYE_INNER_z', 
             'RIGHT_EYE_x', 'RIGHT_EYE_y', 'RIGHT_EYE_z','RIGHT_EYE_OUTER_x', 'RIGHT_EYE_OUTER_y', 'RIGHT_EYE_OUTER_z','LEFT_EAR_x', 
             'LEFT_EAR_y', 'LEFT_EAR_z', 'RIGHT_EAR_x', 'RIGHT_EAR_y','RIGHT_EAR_z', 'MOUTH_LEFT_x', 'MOUTH_LEFT_y', 'MOUTH_LEFT_z',
             'MOUTH_RIGHT_x', 'MOUTH_RIGHT_y', 'MOUTH_RIGHT_z', 'LEFT_SHOULDER_x','LEFT_SHOULDER_y', 'LEFT_SHOULDER_z', 'RIGHT_SHOULDER_x',
             'RIGHT_SHOULDER_y', 'RIGHT_SHOULDER_z', 'LEFT_ELBOW_x', 'LEFT_ELBOW_y','LEFT_ELBOW_z', 'RIGHT_ELBOW_x', 'RIGHT_ELBOW_y', 
             'RIGHT_ELBOW_z','LEFT_WRIST_x', 'LEFT_WRIST_y', 'LEFT_WRIST_z', 'RIGHT_WRIST_x','RIGHT_WRIST_y', 'RIGHT_WRIST_z', 'LEFT_PINKY_x', 
             'LEFT_PINKY_y','LEFT_PINKY_z', 'RIGHT_PINKY_x', 'RIGHT_PINKY_y', 'RIGHT_PINKY_z','LEFT_INDEX_x', 'LEFT_INDEX_y', 'LEFT_INDEX_z',
             'RIGHT_INDEX_x','RIGHT_INDEX_y', 'RIGHT_INDEX_z', 'LEFT_THUMB_x', 'LEFT_THUMB_y','LEFT_THUMB_z', 'RIGHT_THUMB_x', 'RIGHT_THUMB_y','RIGHT_THUMB_z']

audio_col = ['F0semitoneFrom27.5Hz_sma3nz_amean','F1amplitudeLogRelF0_sma3nz_amean','F1bandwidth_sma3nz_amean','F1frequency_sma3nz_amean',
             'F2amplitudeLogRelF0_sma3nz_amean','F2bandwidth_sma3nz_amean','F2frequency_sma3nz_amean','F3amplitudeLogRelF0_sma3nz_amean',
             'F3bandwidth_sma3nz_amean','F3frequency_sma3nz_amean','HNRdBACF_sma3nz_amean','alphaRatioV_sma3nz_amean',
             'hammarbergIndexV_sma3nz_amean','jitterLocal_sma3nz_amean','logRelF0-H1-A3_sma3nz_amean','logRelF0-H1-H2_sma3nz_amean',
             'loudness_sma3_amean','mfcc1_sma3_amean','mfcc2_sma3_amean','mfcc3_sma3_amean','mfcc4_sma3_amean','shimmerLocaldB_sma3nz_amean',
             'slopeV0-500_sma3nz_amean','slopeV500-1500_sma3nz_amean','spectralFlux_sma3_amean']
