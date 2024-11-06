import json
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import shutil

def load_video_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        video_metadata = json.load(f)
    return {'/'+k.split('/')[-1]: v for k, v in video_metadata.items()}

def extract_gesture_images(transcript_path, video_metadata):
    video_name = os.path.basename(transcript_path).replace(".json", "")

    save_dir = os.path.join(os.path.dirname(transcript_path).replace("transcripts", "gestures"), video_name)
    if os.path.exists(os.path.join(save_dir, "txt_img_paths.json")):
        raise FileExistsError(f"Gesture images for {video_name} already exist")
    elif os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    os.makedirs(save_dir, exist_ok=True)

    video_path = transcript_path.replace("transcripts", "video_clips").replace("json", "mp4")
    inv_timestamps = video_metadata['/' + video_name]['timestamp_inv']

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
            frames.append({'time': int(curr_frame // hop)+1, 'frame': frame})
        curr_frame += 1

    cap.release()

    try:
        data = json.load(open(transcript_path, "r", encoding='utf-8'))
    except:
        data = json.load(open(transcript_path, "r", encoding='cp949'))
    frame_times = [frame['time'] for frame in frames]

    save_dir = os.path.join(os.path.dirname(transcript_path).replace("transcripts", "gestures"), video_name)
    os.makedirs(save_dir, exist_ok=True)

    txt_img_paths = []
    id = 0
    for segment in data["transcription"]["segments"]:
        if inv_timestamps:
            segment_start = segment["start"]
            segment_end = segment["end"]
            skip_segment = False
        
            for st, et in inv_timestamps:
                if (segment_start <= et/1000) and (segment_end >= st/1000):
                    skip_segment = True
                    break

            if skip_segment:
                continue 

        for word_info in segment["words"]:
            start_time, end_time, text = word_info["start"], word_info["end"], word_info["text"]
            closest_index = int(np.round(start_time)) if np.round(start_time) <= end_time else int(start_time)
            try:
                frame = frames[frame_times.index(closest_index)]["frame"]
            except:
                continue
            gesture_fp = os.path.join(save_dir, f"{id}_gesture.jpg")
            cv2.imwrite(gesture_fp, frame)

            txt_img_paths.append({
                "token": text,
                "start": start_time,
                "end": end_time,
                "token_img_path": gesture_fp
            })
            id += 1

    with open(os.path.join(save_dir, "txt_img_paths.json"), "w") as f:
        json.dump(txt_img_paths, f, indent=2)

if __name__ == "__main__":
    video_metadata = load_video_metadata('D:/aphasia/MMATD/data_preprocessing/video_metadata.json')
    transcript_paths = glob('D:/aphasia/dataset/transcripts/*.json')
    for transcript_path in tqdm(transcript_paths, desc="Extracting gesture images"):
        try:
            extract_gesture_images(transcript_path, video_metadata)
        except KeyboardInterrupt:
            break
        except FileExistsError:
            continue
        except Exception as e:
            print(e)
            print(f"Failed to extract gesture images from {transcript_path}")
            continue