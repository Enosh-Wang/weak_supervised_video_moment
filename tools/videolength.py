import pandas as pd

train_df = pd.read_csv('/home/share/wangyunxiao/Charades/origin/Charades/Charades_v1_train.csv')
test_df = pd.read_csv('/home/share/wangyunxiao/Charades/origin/Charades/Charades_v1_test.csv')
video_name = train_df['id']+test_df['id']
video_length = train_df['length']+test_df['length']

video_dict = dict.fromkeys(video_name,video_length)