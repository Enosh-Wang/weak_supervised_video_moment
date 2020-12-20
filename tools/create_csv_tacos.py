import pandas as pd
import math
import random
import json
import h5py
import os

datapath = '/home/share/wangyunxiao/TACoS'
test_caption_path = 'annotations/test.json'
val_caption_path = 'annotations/val.json'
train_caption_path = 'annotations/train.json'

def save_csv(name,caption_data):
    video_list = []
    start_time_list = []
    end_time_list = []
    duration_list = []
    sentences_list = []

    for key,value in caption_data.items():
        video = key
        duration = value['num_frames']
        sentences = value['sentences']
        timestamps = value['timestamps']
        
        for i in range(len(sentences)):
            video_list.append(video)
            duration_list.append(duration)
            sentences_list.append(sentences[i])
            start_time_list.append(timestamps[i][0])
            end_time_list.append(timestamps[i][1])

    df = pd.DataFrame()
    df['video'] = video_list
    df['start_time'] = start_time_list
    df['end_time'] = end_time_list
    df['duration'] = duration_list
    df['description'] = sentences_list

    # 不保留行索引数字
    df = df.sample(frac=1)
    df.to_csv(name,index=False)

with open(os.path.join(datapath,train_caption_path), encoding='utf8') as f:
    train_caption_data = json.load(f)
with open(os.path.join(datapath,val_caption_path), encoding='utf8') as f:
    val_caption_data = json.load(f)
with open(os.path.join(datapath,test_caption_path), encoding='utf8') as f:
    test_caption_data = json.load(f)

save_csv('tacos_train.csv',train_caption_data)
save_csv('tacos_val.csv',val_caption_data)
save_csv('tacos_test.csv',test_caption_data)