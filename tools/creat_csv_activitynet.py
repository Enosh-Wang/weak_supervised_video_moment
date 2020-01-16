import pandas as pd
import math
import random
import json
import h5py
import os

datapath = '/home/share/wangyunxiao/ActivityNet'
feature_path = 'sub_activitynet_v1-3.c3d.hdf5'
caption_path = 'caption/val_2.json'

def save_csv(name,caption_data,video_feature):
    video_list = []
    start_time_list = []
    end_time_list = []
    duration_list = []
    sentences_list = []
    start_segment_list = []
    end_segment_list = []

    for key,value in caption_data.items():
        video = key
        duration = value['duration']
        sentences = value['sentences']
        timestamps = value['timestamps']
        feats = video_feature[video]['c3d_features']
        fps = feats.shape[0] / duration
        
        for i in range(len(sentences)):
            video_list.append(video)
            duration_list.append(duration)
            sentences_list.append(sentences[i])
            start_time_list.append(timestamps[i][0])
            end_time_list.append(timestamps[i][1])

            start_frame = int(fps * timestamps[i][0])
            end_frame = int(fps * timestamps[i][1])
            if end_frame >= feats.shape[0]:
                end_frame = feats.shape[0] - 1
            if start_frame > end_frame:
                start_frame = end_frame
            assert start_frame <= end_frame
            assert 0 <= start_frame < feats.shape[0]
            assert 0 <= end_frame < feats.shape[0]
            
            start_segment_list.append(start_frame)
            end_segment_list.append(end_frame)

    df = pd.DataFrame()
    df['video'] = video_list
    df['start_time'] = start_time_list
    df['end_time'] = end_time_list
    df['duration'] = duration_list
    df['description'] = sentences_list
    df['start_segment'] = start_segment_list
    df['end_segment'] = end_segment_list

    # 不保留行索引数字
    df.to_csv(name,index=False)

with open(os.path.join(datapath,caption_path), encoding='utf8') as f:
    caption_data = json.load(f)

save_csv('activitynet_test.csv',caption_data,h5py.File(os.path.join(datapath,feature_path), 'r'))