import pandas as pd
import math
import random
import os
def save_csv(name,data,video_dict):
    video_list = []
    start_time_list = []
    end_time_list = []
    duration_list = []
    description_list = []

    for line in data:
        video_name = line.split(' ')[0]
        start_time = float(line.split(' ')[1])
        end_time = float(line.split('##')[0].split(' ')[2])
        description = line.split('##')[1]
        duration = float(video_dict[video_name])
        end_time = min(end_time,duration)
        
        if start_time < end_time:
            duration_list.append(duration)
            video_list.append(video_name)
            start_time_list.append(start_time)
            end_time_list.append(end_time)
            description_list.append(description)
    
    df = pd.DataFrame()
    df['video'] = video_list
    df['start_time'] = start_time_list
    df['end_time'] = end_time_list
    df['duration'] = duration_list
    df['description'] = description_list
    # 不保留行索引数字
    df.to_csv(name,index=False)


with open("/home/share/wangyunxiao/Charades/caption/charades_sta_train.txt") as f:
    trainval_txt = f.readlines()
with open("/home/share/wangyunxiao/Charades/caption/charades_sta_test.txt") as f:
    test_txt = f.readlines()

num = len(trainval_txt)
train_num = math.floor(num*0.8)
random.shuffle(trainval_txt)
random.shuffle(test_txt)
train_list = trainval_txt[:train_num]
val_list = trainval_txt[train_num:]

train_df = pd.read_csv('/home/share/wangyunxiao/Charades/origin/Charades/Charades_v1_train.csv')
test_df = pd.read_csv('/home/share/wangyunxiao/Charades/origin/Charades/Charades_v1_test.csv')
video_name = list(train_df['id'])+list(test_df['id'])
video_length = list(train_df['length'])+list(test_df['length'])

video_dict = dict(zip(video_name,video_length))
# 解析字段，并保存为csv
save_csv('/home/share/wangyunxiao/Charades/caption/charades_train_valid.csv',train_list,video_dict)
save_csv('/home/share/wangyunxiao/Charades/caption/charades_val_valid.csv',val_list,video_dict)
save_csv('/home/share/wangyunxiao/Charades/caption/charades_test_valid.csv',test_txt,video_dict)






