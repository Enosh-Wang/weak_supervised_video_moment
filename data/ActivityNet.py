import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import numpy as np
import pandas
import scipy.io as sio
import skimage.measure as scikit
import h5py

class ActivityNet(data.Dataset):
    """
    Load precomputed sentences and videos features
    """

    def __init__(self, data_split, data_path, vocab):
        """
        根据选定的数据集，读取csv文件，获取视频列表和标注信息，例如charades_test.csv
        """    
        # Load Vocabulary Wrapper
        self.vocab=vocab
        path=os.path.join(data_path,"caption/activitynet_"+ str(data_split) + ".csv")
        df=pandas.read_csv(path)
        #df_temp = pandas.read_csv(path,dtype={'ID': object})
        # 视频名称列表
        self.videos = df['video']
        # 描述语句列表
        self.description=df['description']
        # 数据集划分
        self.data_split=data_split
        # 数据集地址
        self.data_path=data_path
        feature_path = os.path.join(data_path,'sub_activitynet_v1-3.c3d.hdf5')
        self.feature = h5py.File(feature_path,'r')
        

    def __getitem__(self, index):

        videos=self.videos
        description=self.description
        # load C3D feature 单个视频读取 shape=(n,4096)
        video_feat=self.feature[videos[index]]['c3d_features']
        # 采样non-overlap的clip,16帧一个clip
        video_feat = video_feat[0::2]
        video_feat=scikit.block_reduce(video_feat, block_size=(4, 1), func=np.mean)

        # 数组转成tensor
        video = torch.Tensor(video_feat)
        sentence = description[index]
        vocab = self.vocab

        # Convert sentence (string) to word ids.
        # 分词
        tokens = nltk.tokenize.word_tokenize(str(sentence).lower())
        sentence = []
        # 开头结尾添加分隔符
        sentence.append(vocab('<start>'))
        sentence.extend([vocab(token) for token in tokens])
        sentence.append(vocab('<end>'))
        # 转为tensor
        sentence = torch.Tensor(sentence)
        # video [6,4096] target [12] index 2059 
        return video, sentence, index

    def __len__(self):
        # 长度按照句子数量，其实就是pair的数量
        return len(self.description)