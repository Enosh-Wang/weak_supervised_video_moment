import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import numpy as np
import pandas
import scipy.io as sio
# import skimage.measure as scikit
import pickle

class TACoS(data.Dataset):
    """
    Load precomputed sentences and videos features
    """

    def __init__(self, data_split, data_path, word2vec, vocab):
        """
        根据选定的数据集，读取csv文件，获取视频列表和标注信息，例如charades_test.csv
        """    
        # Load Vocabulary Wrapper
        self.vocab=vocab
        self.word2vec=word2vec
        path=os.path.join(data_path,"caption/tacos_"+ str(data_split) + ".csv")
        df=pandas.read_csv(path)
        # 视频名称列表
        self.videos = df['video']
        # 描述语句列表
        self.description=df['description']
        # 数据集划分
        self.data_split=data_split
        # 数据集地址
        self.data_path=data_path
        feature_path = os.path.join(data_path,'tacos_128.pkl')
        with open(feature_path,'rb') as f:
            self.feature = pickle.load(f)
        

    def __getitem__(self, index):

        video_name=self.videos[index]
        # load C3D feature 单个视频读取
        video_feat=self.feature[video_name]
        video = torch.Tensor(video_feat)
        # 文本特征
        sentence = self.description[index]
        words = nltk.tokenize.word_tokenize(str(sentence).lower())
        sentence = np.asarray([self.word2vec[word] for word in words if word in self.word2vec])
        word_id = np.asarray([self.vocab(word) for word in words if word in self.word2vec])
        sentence = torch.Tensor(sentence)
        word_id = torch.Tensor(word_id)
        return video, sentence, index, video_name, word_id

    def __len__(self):
        # 长度按照句子数量，其实就是pair的数量
        return len(self.description)