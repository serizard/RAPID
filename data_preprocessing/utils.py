import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


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
    
    cls_token = np.random.normal(0, 0.02, (1, feature_dim))
    eos_token = np.random.normal(0, 0.02, (1, feature_dim))
    
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


def create_feature_adjacency_matrices(dfs, disfluency_tokens, video_col, audio_col, std_multiplier=1.5, base_weight=0.3):
    """새로운 방식의 인접 행렬 생성"""
    n_samples = len(dfs)
    n_tokens = len(disfluency_tokens)
    
    # 전체 데이터셋에 대한 co-occurrence 통계 계산
    total_co_occurrences = defaultdict(int)
    total_counts = defaultdict(int)
    
    print("Calculating global co-occurrence statistics...")
    for df in tqdm(dfs):
        # 각 토큰의 출현 위치 확인
        token_positions = defaultdict(list)
        for i, token in enumerate(df['token'].values):
            if token.lower() in disfluency_tokens:
                token_positions[token.lower()].append(i)
        
        # 제스처/오디오 변화 계산
        gesture_changes = calculate_gesture_changes(df, video_col, std_multiplier)
        audio_changes = calculate_audio_changes(df, audio_col)
        
        # Co-occurrence 계산
        for token, positions in token_positions.items():
            for pos in positions:
                # 해당 토큰 주변 ±2 프레임 검사
                start_idx = max(0, pos-2)
                end_idx = min(len(df), pos+3)
                window_gesture = gesture_changes[start_idx:end_idx]
                window_audio = audio_changes[start_idx:end_idx]
                
                if np.any(window_gesture):
                    total_co_occurrences[('gesture', token)] += 1
                if np.any(window_audio):
                    total_co_occurrences[('audio', token)] += 1
                    
                total_counts[token] += 1
    
    # 정규화된 co-occurrence 가중치 계산
    weights = {}
    for (modality, token), count in total_co_occurrences.items():
        if total_counts[token] > 0:
            weights[(modality, token)] = count / total_counts[token]
    
    # 인접 행렬 생성
    print("\nCreating adjacency matrices...")
    adj_matrices = []
    
    for df in tqdm(dfs):
        seq_length = len(df)
        # 시퀀스 길이 x 토큰 수 크기의 인접 행렬
        adj_matrix = np.zeros((seq_length, n_tokens))
        
        # # 각 프레임에서의 변화 감지
        # gesture_changes = calculate_gesture_changes(df, video_col, std_multiplier)
        # audio_changes = calculate_audio_changes(df, audio_col)
        
        # 각 토큰의 출현과 변화를 연결
        for i, token in enumerate(df['token'].values):
            if token.lower() in disfluency_tokens:
                token_idx = disfluency_tokens.index(token.lower())
                
                # 기본 가중치 할당
                adj_matrix[i, token_idx] = base_weight
                
                # 현재 프레임 주변 ±2 프레임 검사
                # start_idx = max(0, i-2)
                # end_idx = min(seq_length, i+3)
                # window_gesture = gesture_changes[start_idx:end_idx]
                # window_audio = audio_changes[start_idx:end_idx]
                
                # 제스처나 오디오 변화가 있는 경우 추가 가중치 할당
                additional_weight = weights.get(('gesture', token.lower()), 0) + weights.get(('audio', token.lower()), 0) - 1.7
                additional_weight *= 100
                # if np.any(window_gesture):
                #     additional_weight += weights.get(('gesture', token.lower()), 0)
                # if np.any(window_audio):
                #     additional_weight += weights.get(('audio', token.lower()), 0)
                
                # 기존 가중치에 추가 가중치를 더함
                adj_matrix[i, token_idx] += additional_weight
        
        adj_matrices.append(adj_matrix)
    
    final_adj_matrices = np.stack(adj_matrices)
    return final_adj_matrices



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
