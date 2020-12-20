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
from tools.util import get_match_map, iou_with_anchors, get_mask, get_mask_spare
#from transformers import BertTokenizer

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
        #　时间戳
        self.start_time = df['start_time']
        self.end_time = df['end_time']
        self.duration = df['duration']
        # 数据集划分
        self.data_split=data_split
        # 数据集地址
        self.data_path=data_path
        # 读取特征
        feature_path = os.path.join(data_path,'charades_31.pkl') # charades_n20_mean
        with open(feature_path,'rb') as f:
            self.feature = pickle.load(f)

    def __getitem__(self, index):

        video_name=self.videos[index]
        # load C3D feature 单个视频读取
        video_feat=self.feature[video_name]
        #video_feat=scikit.block_reduce(video_feat, block_size=(2, 1), func=np.mean)
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


class Charades_Bert(data.Dataset):
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
        #　时间戳
        self.start_time = df['start_time']
        self.end_time = df['end_time']
        self.duration = df['duration']
        # 数据集划分
        self.data_split=data_split
        # 数据集地址
        self.data_path=data_path
        # 读取特征
        feature_path = os.path.join(data_path,'charades_n20_mean.pkl')
        with open(feature_path,'rb') as f:
            self.feature = pickle.load(f)

        self.tokenizer = BertTokenizer.from_pretrained('/home/share/wangyunxiao/BERT/bert-base-uncased-vocab.txt')

    def __getitem__(self, index):

        video_name=self.videos[index]
        # load C3D feature 单个视频读取
        video_feat=self.feature[video_name]
        #video_feat=scikit.block_reduce(video_feat, block_size=(2, 1), func=np.mean)

        # 处理文本特征
        sentence = self.description[index]
        encoded_inputs = self.tokenizer(sentence)
        word_id = encoded_inputs['input_ids']
        segment = encoded_inputs['token_type_ids']
        input_mask = encoded_inputs['attention_mask']

        tmp_start = max(min(1, self.start_time[index] / self.duration[index]), 0)
        tmp_end = max(min(1, self.end_time[index] / self.duration[index]), 0)

        match_map = get_match_map(20,0,0.5)
        mask = get_mask(20,0,0.5)

        tmp_gt_iou_map = iou_with_anchors(match_map[:, 0], match_map[:, 1], tmp_start, tmp_end)

        tmp_gt_iou_map = np.reshape(tmp_gt_iou_map, [-1, 20])
        
        gt_map = tmp_gt_iou_map*mask

        return video_feat, index, word_id, gt_map, segment, input_mask

    def __len__(self):
        # 长度按照句子数量，其实就是pair的数量
        return len(self.description)


	
