import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

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


def get_labels(aphasia_type):   
    aphasia_type = aphasia_type.lower()
    type = ""
    ct_label = wab_label = type_label = flu_label = com_label = 0
    
    if aphasia_type in ['control', 'notaphasicbywab']:
        type, status = "Control", 'Control'
        
    elif aphasia_type in ['anomic', 'conduction']:
        type = aphasia_type.capitalize()
        status = 'Aphasia'
        ct_label = wab_label = type_label = flu_label = 1
        
    elif aphasia_type in ['wernicke', 'transsensory']:
        type = "Wernicke" if aphasia_type == 'wernicke' else "Trans. Sensory"
        status = 'Aphasia'
        ct_label = wab_label = type_label = flu_label = com_label = 1
        wab_label = 2
        type_label = 2
        
    elif aphasia_type in ['broca', 'transmotor']:
        type = "Broca" if aphasia_type == 'broca' else "Trans. Motor"
        status = 'Aphasia'
        ct_label = wab_label = type_label = 1
        wab_label = 3
        type_label = 3
        
    elif aphasia_type in ['global', 'isolation']:
        type = "Global" if aphasia_type == 'global' else "Isolation"
        status = 'Aphasia'
        ct_label = wab_label = type_label = com_label = 1
        wab_label = 4
        type_label = 3
    
    else:
        return None, None, None, None, None, None, None

    return type, status, ct_label, wab_label, type_label, flu_label, com_label

def add_special_tokens_to_features(features):
    _, feature_dim = features.shape
    cls_token = np.ones((1, feature_dim))
    eos_token = -1 * np.ones((1, feature_dim))
    return np.concatenate([cls_token, features, eos_token], axis=0)

def calculate_gesture_changes(df, video_col, threshold_multiplier=1.5):
    changes = np.zeros(len(df))
    
    for col in video_col:
        if col in df.columns:
            diff = np.abs(np.diff(df[col].values, prepend=df[col].values[0]))
            threshold = np.std(diff) * threshold_multiplier
            changes += (diff > threshold).astype(float)
    
    return (changes > 0).astype(bool)

def calculate_audio_changes(df, audio_col, z_score_threshold=1.0):
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
    def safe_correlation(x, y):
        if len(x) < 2 or len(y) < 2:
            return 0.0
        if np.all(x == x[0]) or np.all(y == y[0]):
            return 0.0
        try:
            corr = np.corrcoef(x, y)[0,1]
            return 0.0 if np.isnan(corr) else corr
        except:
            return 0.0

    def calculate_temporal_correlation(token_features, modal_features, window_size=3):
        if len(token_features) < window_size:
            return 0.0
            
        correlations = []
        for i in range(len(token_features) - window_size + 1):
            token_window = token_features[i:i + window_size]
            modal_window = modal_features[i:i + window_size]
            
            modal_abs = np.abs(modal_window)
            modal_change = np.zeros(len(modal_window))
            
            for idx in range(len(modal_window)):
                row = modal_abs[idx]
                if not np.all(np.isnan(row)):
                    modal_change[idx] = np.nanmean(row)
                else:
                    modal_change[idx] = 0.0
            
            token_change = np.sum(token_window, axis=1)
            
            if not np.any(np.isnan(token_change)) and not np.any(np.isnan(modal_change)):
                corr = safe_correlation(token_change, modal_change)
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0

    def safe_mean(x):
        return np.nanmean(x) if len(x) > 0 and not np.all(np.isnan(x)) else 0.0
    
    def safe_std(x):
        return np.nanstd(x) if len(x) > 0 and not np.all(np.isnan(x)) else 1.0

    def calculate_gesture_significance(gesture_features):
        if len(gesture_features) < 2:
            return 0.0
        try:
            movement = np.diff(gesture_features, axis=0)
            magnitude = np.linalg.norm(movement, axis=1)
            magnitude = magnitude[~np.isnan(magnitude)]
            if len(magnitude) < 2:
                return 0.0
            mean_magnitude = safe_mean(magnitude)
            std_magnitude = safe_std(magnitude)
            return mean_magnitude * (1 + std_magnitude)
        except:
            return 0.0

    def calculate_audio_significance(audio_features):
        if len(audio_features) < 2:
            return 0.0
        try:
            clean_features = audio_features[~np.any(np.isnan(audio_features), axis=1)]
            if len(clean_features) < 2:
                return 0.0
            intensity = np.linalg.norm(clean_features, axis=1)
            mean_intensity = safe_mean(intensity)
            std_intensity = safe_std(intensity)
            return mean_intensity * (1 + std_intensity)
        except:
            return 0.0

    n_samples = len(dfs)
    n_tokens = len(disfluency_tokens)
    adj_matrices = []
    
    print("Computing enhanced feature correlations...")
    for df in tqdm(dfs):
        token_matrix = np.zeros((len(df), n_tokens))
        for i, token in enumerate(df['token'].values):
            if token.lower() in disfluency_tokens:
                token_idx = disfluency_tokens.index(token.lower())
                token_matrix[i, token_idx] = 1
        
        gesture_features = df[video_col].values
        audio_features = df[audio_col].values
        
        token_weights = np.zeros(n_tokens)
        for j, token in enumerate(disfluency_tokens):
            token_mask = token_matrix[:, j] == 1
            if np.any(token_mask):
                temporal_corr_gesture = calculate_temporal_correlation(
                    token_matrix[:, [j]], gesture_features)
                temporal_corr_audio = calculate_temporal_correlation(
                    token_matrix[:, [j]], audio_features)
                
                gesture_sig = calculate_gesture_significance(gesture_features[token_mask])
                audio_sig = calculate_audio_significance(audio_features[token_mask])
                
                gesture_weight = temporal_corr_gesture * gesture_sig
                audio_weight = temporal_corr_audio * audio_sig
                
                total_weight = gesture_weight + audio_weight
                token_weights[j] = total_weight if total_weight != 0 else 0.0
        
        adj_matrix = token_matrix * token_weights
        adj_matrices.append(adj_matrix)
    
    return np.stack(adj_matrices)