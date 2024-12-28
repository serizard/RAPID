import pandas as pd
import mediapipe as mp
import cv2
import os
import numpy as np
import time
from glob import glob
from multiprocessing import Pool, Manager
import warnings
warnings.filterwarnings("ignore")

def process_single_video(args):
    st = time.time()
    txt_video_path, pose_file_list, video_col = args
    mp_holistic = mp.solutions.holistic
    
    new_df = pd.DataFrame()
    new_file_name = txt_video_path.replace('text_bind', 'pose_bind')
    
    if new_file_name in pose_file_list:
        return new_file_name
        
    new_path = os.path.dirname(new_file_name)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        
    txt_df = pd.read_json(txt_video_path)
    
    with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as holistic:
        for token_img_path in txt_df['token_img_path'].tolist():
            image = cv2.imread(token_img_path)
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            dic = {'token_img_path': token_img_path}
            if results.pose_landmarks:
                for mark, data_point in zip(mp_holistic.PoseLandmark, results.pose_landmarks.landmark):
                    dic[f'{mark.name}_x'] = data_point.x
                    dic[f'{mark.name}_y'] = data_point.y
                    dic[f'{mark.name}_z'] = data_point.z
            else:
                for mark in mp_holistic.PoseLandmark:
                    dic[f'{mark.name}_x'] = np.nan
                    dic[f'{mark.name}_y'] = np.nan
                    dic[f'{mark.name}_z'] = np.nan
                    
            new_df = pd.concat([new_df, pd.json_normalize(dic)], ignore_index=True)
    
    new_df = new_df[['token_img_path'] + video_col]
    new_df.to_json(new_file_name, orient='records')
    et = time.time()
    print(f'Processed {txt_video_path} in {et-st:.2f} seconds')
    return new_file_name

def main():
    # Data loading
    ex_df = pd.read_csv('/workspace/dataset/paths.csv')
    pose_file_list = glob('/workspace/dataset/tokens/pose_bind/*/*.json')
    
    video_col = ['NOSE_x', 'NOSE_y', 'NOSE_z', 'LEFT_EYE_INNER_x', 'LEFT_EYE_INNER_y', 'LEFT_EYE_INNER_z', 
                'LEFT_EYE_x', 'LEFT_EYE_y', 'LEFT_EYE_z', 'LEFT_EYE_OUTER_x', 'LEFT_EYE_OUTER_y', 'LEFT_EYE_OUTER_z', 
                'RIGHT_EYE_INNER_x', 'RIGHT_EYE_INNER_y', 'RIGHT_EYE_INNER_z', 'RIGHT_EYE_x', 'RIGHT_EYE_y', 
                'RIGHT_EYE_z', 'RIGHT_EYE_OUTER_x', 'RIGHT_EYE_OUTER_y', 'RIGHT_EYE_OUTER_z', 'LEFT_EAR_x', 
                'LEFT_EAR_y', 'LEFT_EAR_z', 'RIGHT_EAR_x', 'RIGHT_EAR_y', 'RIGHT_EAR_z', 'MOUTH_LEFT_x', 
                'MOUTH_LEFT_y', 'MOUTH_LEFT_z', 'MOUTH_RIGHT_x', 'MOUTH_RIGHT_y', 'MOUTH_RIGHT_z', 
                'LEFT_SHOULDER_x', 'LEFT_SHOULDER_y', 'LEFT_SHOULDER_z', 'RIGHT_SHOULDER_x', 'RIGHT_SHOULDER_y', 
                'RIGHT_SHOULDER_z', 'LEFT_ELBOW_x', 'LEFT_ELBOW_y', 'LEFT_ELBOW_z', 'RIGHT_ELBOW_x', 
                'RIGHT_ELBOW_y', 'RIGHT_ELBOW_z', 'LEFT_WRIST_x', 'LEFT_WRIST_y', 'LEFT_WRIST_z', 
                'RIGHT_WRIST_x', 'RIGHT_WRIST_y', 'RIGHT_WRIST_z', 'LEFT_PINKY_x', 'LEFT_PINKY_y', 
                'LEFT_PINKY_z', 'RIGHT_PINKY_x', 'RIGHT_PINKY_y', 'RIGHT_PINKY_z', 'LEFT_INDEX_x', 
                'LEFT_INDEX_y', 'LEFT_INDEX_z', 'RIGHT_INDEX_x', 'RIGHT_INDEX_y', 'RIGHT_INDEX_z', 
                'LEFT_THUMB_x', 'LEFT_THUMB_y', 'LEFT_THUMB_z', 'RIGHT_THUMB_x', 'RIGHT_THUMB_y', 'RIGHT_THUMB_z']
    
    # Prepare arguments for parallel processing
    args_list = [(path, pose_file_list, video_col) for path in ex_df['txt_img_path'].tolist()]
    # Create processing pool
    num_processes = os.cpu_count() // 2  # Using half of available CPU cores
    
    # Process videos in parallel with tqdm
    with Pool(num_processes) as pool:
        new_pose_files = list(pool.imap(process_single_video, args_list))
    
    # Update DataFrame with new pose files
    ex_df['pose_path'] = new_pose_files
    
    # Save updated DataFrame
    ex_df.to_csv('/workspace/dataset/paths_pose_added.csv', index=False)

if __name__ == '__main__':
    main()