import pandas as pd
import opensmile
from tqdm import tqdm 
import argparse
from pydub import AudioSegment
import os 
from glob import glob
import multiprocessing
import tempfile
import warnings
warnings.filterwarnings("ignore")

ex_df = pd.read_csv('D:/aphasia/dataset/remake_dataset/paths.csv')

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
    num_workers=multiprocessing.cpu_count()*2
)

opensmile_file_list = glob('D:/aphasia/dataset/remake_dataset/tokens/opensmile_bind/*/*.json')

# 임시 파일을 저장할 디렉토리 생성
temp_dir = "temp_wav_files"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

for row1 in tqdm(ex_df.itertuples(), total=len(ex_df)):
    new_file_name = row1.txt_img_path.replace('text_bind', 'opensmile_bind')
    if new_file_name in opensmile_file_list:
        continue

    new_path = os.path.dirname(new_file_name)
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    try:
        txt_df = pd.read_json(row1.txt_img_path)
    except:
        continue

    # MP3를 WAV로 변환
    temp_wav_path = os.path.join(temp_dir, f"temp_{os.path.basename(row1.audio_path)}.wav")
    try:
        audio = AudioSegment.from_mp3(row1.audio_path)
        audio.export(temp_wav_path, format="wav")
        
        zero_point = txt_df.start.iloc[0]
        aud_df = pd.DataFrame()
        
        for row2 in txt_df.itertuples():
            y = smile.process_file(temp_wav_path, start=row2.start-zero_point, end=row2.end-zero_point)
            aud_df = pd.concat([aud_df,y], axis=0)

        # 처리가 끝난 후 임시 WAV 파일 삭제
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

        aud_df.reset_index(drop=True).to_json(new_file_name)
        opensmile_file_list.append(new_file_name)
    except Exception as e:
        print(f"Error processing {row1.audio_path}: {str(e)}")
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        continue

ex_df['opensmile_path'] = opensmile_file_list
ex_df.to_csv('D:/aphasia/dataset/paths_opensmile_added.csv', index=False)
