import pandas as pd
import math
import random

def save_csv(name,data):
    video_list = []
    start_time_list = []
    end_time_list = []
    description_list = []
    start_segment_list = []
    end_segment_list = []

    for line in data:
        video_name = line.split(' ')[0]
        start_time = line.split(' ')[1]
        end_time = line.split('##')[0].split(' ')[2]
        description = line.split('##')[1]
        start_segment = str(math.floor(float(start_time)*24))
        end_segment = str(math.floor(float(end_time)*24))

        video_list.append(video_name)
        start_time_list.append(start_time)
        end_time_list.append(end_time)
        description_list.append(description)
        start_segment_list.append(start_segment)
        end_segment_list.append(end_segment)
    
    df = pd.DataFrame()
    df['video'] = video_list
    df['start_time'] = start_time_list
    df['end_time'] = end_time_list
    df['description'] = description_list
    df['start_segment'] = start_segment_list
    df['end_segment'] = end_segment_list
    # 不保留行索引数字
    df.to_csv(name,index=False)


with open("charades_sta_train.txt") as f:
    # 读取txt文件，划分train、val
    txt = f.readlines()
    num = len(txt)
    train_num = math.floor(num*0.8)
    train_list = txt[:train_num]
    val_list = txt[train_num:]
    
    # 解析字段，并保存为csv
    save_csv('charades_train.csv',train_list)
    save_csv('charades_val.csv',val_list)






