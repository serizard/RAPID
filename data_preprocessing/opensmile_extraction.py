
import pandas as pd
import opensmile
from tqdm import tqdm 
import argparse
from pydub import AudioSegment
import os 

from glob import glob

import multiprocessing


import warnings
warnings.filterwarnings("ignore")

ex_df= pd.read_csv('D:/aphasia/dataset/remake_dataset/paths.csv')

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
    num_workers=multiprocessing.cpu_count()*2
)

opensmile_file_list = glob('D:/aphasia/dataset/remake_dataset/tokens/opensmile_bind/*/*.json')
for row1 in tqdm(ex_df.itertuples(),total=len(ex_df)):
    new_file_name = row1.txt_img_path.replace('text_bind','opensmile_bind')
    if new_file_name in opensmile_file_list:
        continue

    new_path = os.path.dirname(new_file_name)
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    try:
        txt_df = pd.read_json(row1.txt_img_path)
    except:
        continue

    zero_point = txt_df.start.iloc[0]
    aud_df = pd.DataFrame()
    for row2 in txt_df.itertuples():
        y = smile.process_file(row1.audio_path, start=row2.start-zero_point, end=row2.end-zero_point)
        aud_df = pd.concat([aud_df,y],axis=0)

    aud_df.reset_index(drop=True).to_json(new_file_name)
    opensmile_file_list.append(new_file_name)

ex_df['opensmile_path'] = opensmile_file_list

ex_df.to_csv('D:/aphasia/dataset/paths_opensmile_added.csv',index=False)