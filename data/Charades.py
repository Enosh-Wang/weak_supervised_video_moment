import torch
import torch.utils.data as data
import os
import nltk
import numpy as np
import pandas
import pickle

class Charades(data.Dataset):

    def __init__(self, data_split, data_path, word2vec):
        
        self.word2vec = word2vec
        path=data_path+"/caption/charades_"+ str(data_split) + ".csv"
        df=pandas.read_csv(path)
        # 视频名称列表
        self.videos = df['video']
        # 描述语句列表
        self.description=df['description']
        #　时间戳
        self.start_time = df['start_time']
        self.end_time = df['end_time']
        self.duration = df['duration']
        # 读取特征
        feature_path = os.path.join(data_path,'charades_31.pkl') # charades_n20_mean
        with open(feature_path,'rb') as f:
            self.feature = pickle.load(f)

    def __getitem__(self, index):

        video_name=self.videos[index]
        # load C3D feature 单个视频读取
        video_feat=self.feature[video_name]
        video = torch.Tensor(video_feat)
        # 处理文本特征
        sentence = self.description[index]
        words = nltk.tokenize.word_tokenize(str(sentence).lower())
        sentence = np.asarray([self.word2vec[word] for word in words if word in self.word2vec])
        sentence = torch.Tensor(sentence)

        return video, sentence, index, video_name

    def __len__(self):
        # 长度按照句子数量，其实就是pair的数量
        return len(self.description)



