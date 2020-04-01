import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import numpy as np
import pandas
import scipy.io as sio
import skimage.measure as scikit
import pickle
import json

class Charades(data.Dataset):
    """
    Load precomputed sentences and video features
    """

    def __init__(self, data_split, data_path, word2vec, vocab):
        """
        根据选定的数据集，读取csv文件，获取视频列表和标注信息，例如charades_test.csv
        """
        self.vocab=vocab
        self.word2vec = word2vec
        path=data_path+"/caption/charades_"+ str(data_split) + ".csv"
        df=pandas.read_csv(path)
        # 视频名称列表
        self.videos = df['video']
        # 描述语句列表
        self.description=df['description']
        # 数据集划分
        self.data_split=data_split
        # 数据集地址
        self.data_path=data_path
        # 读取特征
        feature_path = os.path.join(data_path,'charades_n20_mean.pkl')
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
        word_id = np.asarray([self.vocab(word) for word in words if word in self.word2vec])
        sentence = torch.Tensor(sentence)
        word_id = torch.Tensor(word_id)
        return video, sentence, index, video_name, word_id

    def __len__(self):
        # 长度按照句子数量，其实就是pair的数量
        return len(self.description)

class CharadesGCN(data.Dataset):
    """
    Load precomputed sentences and video features
    """

    def __init__(self, data_split, data_path, word2vec):
        """
        根据选定的数据集，读取csv文件，获取视频列表和标注信息，例如charades_test.csv
        """
        self.word2vec = word2vec
        path=data_path+"/caption/charades_"+ str(data_split) + "_nlp.csv"
        df=pandas.read_csv(path)
        # 视频名称列表
        self.videos = df['video']
        # 描述语句列表
        self.words = df['words']
        self.id2pos = df['id2pos']
        self.sentence_mat = df['mat']
        # 数据集划分
        self.data_split=data_split
        # 数据集地址
        self.data_path=data_path
        # 读取特征
        feature_path = os.path.join(data_path,'charades_n20.pkl')
        with open(feature_path,'rb') as f:
            self.feature = pickle.load(f)
        
        with open('/home/share/wangyunxiao/ContrastiveLosses4VRD/Outputs/all_matrix_file_100_30.json','r') as f:
            self.video_mat = json.load(f)

    def __getitem__(self, index):

        video_name = self.videos[index]
        # load C3D feature 单个视频读取
        video_feat = self.feature[video_name]
        video = torch.Tensor(video_feat)
        video_mat = np.asarray(self.video_mat[video_name])
        video_mat = torch.Tensor(video_mat)
        # 处理文本特征
        words = self.words[index]
        words = np.asarray([self.word2vec[word] for word in words]) # if word in self.word2vec
        sentence = torch.Tensor(words)

        id2pos = np.asarray(self.id2pos[index],dtype=np.int)
        sentence_mat = np.asarray(self.sentence_mat[index])
        # 三者单标的分词数目应相等
        assert id2pos.shape[0] == sentence_mat.shape[0] == words.shape[0]
        
        return video, video_mat, sentence, id2pos, sentence_mat, index, video_name

    def __len__(self):
        # 长度按照句子数量，其实就是pair的数量
        return len(self.words)





	
