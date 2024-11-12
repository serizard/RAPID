import pandas as pd
from glob import glob
import os
import numpy as np
import json
import sys
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import numpy as np
from utils import get_labels, add_special_tokens_to_features, create_feature_adjacency_matrices, audio_col, video_col


def make_chunks(args):
    with open('D:/aphasia/MMATD/data_preprocessing/metadata.json','r') as f:
        previous_metadata = json.load(f)
        metadata = {}
        for k,v in previous_metadata.items():
            metadata[k.split('/')[1]] = v

    dataset_chunk = {"user_name": {}, 'status': {}, 'chunk_id': {}, 
                    "type": {}, "sex": {}, "asr_body_pre": {}, 
                    "ct_label": {}, "wab_label": {}, "type_label": {},
                    "flu_label": {}, "com_label": {}, "data_id": {},
                    "duration": {}}

    dirs = glob('D:/aphasia/dataset/tokens/text_bind/*')
    dfs = []
    audio_feats =[]


    data_id = -1
    for dir in tqdm(dirs):
        # aling the features
        txt_img_path = os.path.join(dir,'txt_img_paths.json')
        opensmile_path = os.path.join(dir.replace('text_bind','opensmile_bind'),'txt_img_paths.json')
        pose_path = os.path.join(dir.replace('text_bind','pose_bind'),'txt_img_paths.json')

        row = {}
        row['txt_img_path'] = txt_img_path 
        row['opensmile_path'] = opensmile_path
        row['pose_path'] = pose_path

        df_txt = pd.read_json(row['txt_img_path'])
        df_aud = pd.read_json(row['opensmile_path'])[audio_col].interpolate()
        df_vid = pd.read_json(row['pose_path'])[video_col].interpolate()

        df_txt = pd.concat([df_txt,df_vid],axis=1)
        df_txt = pd.concat([df_txt,df_aud],axis=1)

        fn = df_txt.iloc[0]['token_img_path'].split('\\')[1]
        metadata_info = metadata[fn]
        aphasia_type, status, ct_label, wab_label, type_label, flu_label, com_label = get_labels(metadata_info['aphasia_type'])

        if aphasia_type is None:
            continue
        # extract information for dataset_chunk
        for i in range(0, len(df_txt), args.chunk_size):
            with open(f'D:/aphasia/dataset/transcripts/{fn}.json', 'r', encoding='utf-8') as f:
                transcript = json.load(f)
            try:
                if i + args.chunk_size <= len(df_txt):
                    dfs.append(df_txt.iloc[i:i+args.chunk_size])
                else:
                    continue
                data_id += 1
                dataset_chunk['duration'][data_id] = transcript['chunks'][i//args.chunk_size]['end'] - transcript['chunks'][i//args.chunk_size]['start']
                asr_body_pre = ' '.join(df_txt.iloc[i:i+args.chunk_size]['token'])

                dataset_chunk['data_id'][data_id] = data_id
                dataset_chunk['user_name'][data_id] = fn

                dataset_chunk['chunk_id'][data_id] = i % args.chunk_size
                dataset_chunk['asr_body_pre'][data_id] = ' '.join(df_txt.iloc[i:i+args.chunk_size]['token']).replace('\u2014', ' ')
            

                dataset_chunk['sex'][data_id] = metadata_info['sex']
                dataset_chunk['data_id'][data_id] = data_id

                dataset_chunk['status'][data_id] = status
                dataset_chunk['type'][data_id] = aphasia_type
                dataset_chunk['ct_label'][data_id] = ct_label
                dataset_chunk['wab_label'][data_id] = wab_label
                dataset_chunk['type_label'][data_id] = type_label
                dataset_chunk['flu_label'][data_id] = flu_label
                dataset_chunk['com_label'][data_id] = com_label
            except:
                continue

    with open(f'D:/aphasia/MMATD/dataset/dataset_chunk{args.chunk_size}.json','w') as f:
        json.dump(dataset_chunk,f, indent=4)


    audio_feats_with_additional_tokens = []
    video_feats_with_additional_tokens = []
    audio_feats = []
    video_feats = []

    for df in tqdm(dfs):
        audio_feat = np.array(df[audio_col].values)
        video_feat = np.array(df[video_col].values)
        audio_feats.append(audio_feat)
        video_feats.append(video_feat)

        audio_feats_with_additional_tokens.append(add_special_tokens_to_features(audio_feat))
        video_feats_with_additional_tokens.append(add_special_tokens_to_features(video_feat))

    audio_feats = np.stack(audio_feats)
    video_feats = np.stack(video_feats)
    audio_feats_with_additional_tokens = np.stack(audio_feats_with_additional_tokens)
    video_feats_with_additional_tokens = np.stack(video_feats_with_additional_tokens)


    np.save(f'D:/aphasia/MMATD/dataset/opensmile_chunk{args.chunk_size+2}_feat.npy', audio_feats_with_additional_tokens)
    np.save(f'D:/aphasia/MMATD/dataset/pose_chunk{args.chunk_size+2}_feat.npy', video_feats_with_additional_tokens)
    np.save(f'D:/aphasia/MMATD/dataset/opensmile_chunk{args.chunk_size}_feat.npy', audio_feats)
    np.save(f'D:/aphasia/MMATD/dataset/pose_chunk{args.chunk_size}_feat.npy', video_feats)

    Type_Control = len([v for v in dataset_chunk['type'].values() if v == 'Control'])
    Type_Anomic = len([v for v in dataset_chunk['type'].values() if v == 'Anomic'])
    Type_Conduction = len([v for v in dataset_chunk['type'].values() if v == 'Conduction'])
    Type_Wernicke = len([v for v in dataset_chunk['type'].values() if v == 'Wernicke'])
    Type_Sensory = len([v for v in dataset_chunk['type'].values() if v == 'Trans. Sensory'])
    Type_Broca = len([v for v in dataset_chunk['type'].values() if v == 'Broca'])
    Type_Motor = len([v for v in dataset_chunk['type'].values() if v == 'Trans. Motor'])

    Subjects_Control = len(np.unique([v for k, v in dataset_chunk['user_name'].items() if dataset_chunk['type'][k] == 'Control']))
    Subjects_Anomic = len(np.unique([v for k, v in dataset_chunk['user_name'].items() if dataset_chunk['type'][k] == 'Anomic']))
    Subjects_Conduction = len(np.unique([v for k, v in dataset_chunk['user_name'].items() if dataset_chunk['type'][k] == 'Conduction']))
    Subjects_Wernicke = len(np.unique([v for k, v in dataset_chunk['user_name'].items() if dataset_chunk['type'][k] == 'Wernicke']))
    Subjects_Sensory = len(np.unique([v for k, v in dataset_chunk['user_name'].items() if dataset_chunk['type'][k] == 'Trans. Sensory']))
    Subjects_Broca = len(np.unique([v for k, v in dataset_chunk['user_name'].items() if dataset_chunk['type'][k] == 'Broca']))
    Subjects_Motor = len(np.unique([v for k, v in dataset_chunk['user_name'].items() if dataset_chunk['type'][k] == 'Trans. Motor']))

    Duration_Control = np.mean([v for k, v in dataset_chunk['duration'].items() if dataset_chunk['type'][k] == 'Control'])
    Duration_Anomic = np.mean([v for k, v in dataset_chunk['duration'].items() if dataset_chunk['type'][k] == 'Anomic'])
    Duration_Conduction = np.mean([v for k, v in dataset_chunk['duration'].items() if dataset_chunk['type'][k] == 'Conduction'])
    Duration_Wernicke = np.mean([v for k, v in dataset_chunk['duration'].items() if dataset_chunk['type'][k] == 'Wernicke'])
    Duration_Sensory = np.mean([v for k, v in dataset_chunk['duration'].items() if dataset_chunk['type'][k] == 'Trans. Sensory'])
    Duration_Broca = np.mean([v for k, v in dataset_chunk['duration'].items() if dataset_chunk['type'][k] == 'Broca'])
    Duration_Motor = np.mean([v for k, v in dataset_chunk['duration'].items() if dataset_chunk['type'][k] == 'Trans. Motor'])
    Duration_Total = np.mean([v for k, v in dataset_chunk['duration'].items()])      

    with open(f'D:/aphasia/MMATD/dataset/dataset_chunk{args.chunk_size}_stats.txt', 'w') as f:
        f.write(f'Aphasia Type\n')
        f.write(f'Control: {Type_Control}\n')
        f.write(f'Anomic: {Type_Anomic}\n')
        f.write(f'Conduction: {Type_Conduction}\n')
        f.write(f'Wernicke: {Type_Wernicke}\n')
        f.write(f'Sensory: {Type_Sensory}\n')
        f.write(f'Broca: {Type_Broca}\n')
        f.write(f'Motor: {Type_Motor}\n')

        f.write(f'\nSubjects\n')
        f.write(f'Control: {Subjects_Control}\n')
        f.write(f'Anomic: {Subjects_Anomic}\n')
        f.write(f'Conduction: {Subjects_Conduction}\n')
        f.write(f'Wernicke: {Subjects_Wernicke}\n')
        f.write(f'Sensory: {Subjects_Sensory}\n')
        f.write(f'Broca: {Subjects_Broca}\n')
        f.write(f'Motor: {Subjects_Motor}\n')

        f.write(f'\nDuration\n')
        f.write(f'Control: {Duration_Control}\n')
        f.write(f'Anomic: {Duration_Anomic}\n')
        f.write(f'Conduction: {Duration_Conduction}\n')
        f.write(f'Wernicke: {Duration_Wernicke}\n')
        f.write(f'Sensory: {Duration_Sensory}\n')
        f.write(f'Broca: {Duration_Broca}\n')
        f.write(f'Motor: {Duration_Motor}\n')
        f.write(f'Total: {Duration_Total}\n')
                    
                                                                                                                                                                                                
    return dfs

def make_adj(args, dfs):
    disfluency_tokens = json.load(open('D:/aphasia/MMATD/dataset/_disfluency_tk_300.json', 'r'))

    adj_matrix = create_feature_adjacency_matrices(
        dfs, 
        disfluency_tokens, 
        video_col, 
        audio_col,
        std_multiplier=args.std_multiplier
    )
    print(f"Adjacency matrix shape: {adj_matrix.shape}")

    np.save(f'D:/aphasia/MMATD/dataset/adj_chunk{args.chunk_size}_300_duration_stdmult{args.std_multiplier}.npy', adj_matrix)



class Args:
    chunk_size = 50
    std_multiplier = 1.5

if __name__ == '__main__':
    args = Args()
    for i1, chunk_size in enumerate([50]):
        args.chunk_size = chunk_size
        dfs = make_chunks(args)
        make_adj(args, dfs)