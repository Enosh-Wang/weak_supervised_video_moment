import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import numpy as np
import pandas
import scipy.io as sio
import skimage.measure as scikit



class Charades(data.Dataset):
    """
    Load precomputed sentences and video features
    """

    def __init__(self, data_split, data_path, vocab):
        """
        根据选定的数据集，读取csv文件，获取视频列表和标注信息，例如charades_test.csv
        """
        self.vocab=vocab
        path=data_path+"/caption/charades_"+ str(data_split) + ".csv"
        df=pandas.read_csv(path)
        df_temp = pandas.read_csv(path,dtype={'ID': object})
        # 视频名称列表
        self.videos = df_temp['video']
        # 描述语句列表
        self.description=df['description']
        # 数据集划分
        self.data_split=data_split
        # 数据集地址
        self.data_path=data_path

    def __getitem__(self, index):

        videos=self.videos
        description=self.description
        # load C3D feature 单个视频读取
        video_feat_file=self.data_path+"/c3d_features/"+str(videos[index])+".mat"
        video_feat_mat = sio.loadmat(video_feat_file)
        video_feat=video_feat_mat['feature']

        # 128 frame features 128帧的滑动窗口 8*16 shape=(2,4096)
        video_feat1=scikit.block_reduce(video_feat, block_size=(8, 1), func=np.mean)
        # 256 frame features 256帧的滑动窗口 16*16 shape=(4,4096)
        video_feat2=scikit.block_reduce(video_feat, block_size=(16, 1), func=np.mean)
        # 添加上下文信息
        
        video_feat_global = np.mean(video_feat,axis=0,keepdims=True)

        feat1_length = len(video_feat1)
        video_feat1_left = np.zeros_like(video_feat1)
        video_feat1_left[0] = video_feat1[0]
        video_feat1_left[range(1,feat1_length)] = video_feat1[range(feat1_length-1)]
        video_feat1_right = np.zeros_like(video_feat1)
        video_feat1_right[-1] = video_feat1[-1]
        video_feat1_right[range(feat1_length-1)] = video_feat1[range(1,feat1_length)]
        video_feat1_global = np.repeat(video_feat_global,feat1_length,axis=0)

        feat2_length = len(video_feat2)
        video_feat2_left = np.zeros_like(video_feat2)
        video_feat2_left[0] = video_feat2[0]
        video_feat2_left[range(1,feat2_length)] = video_feat2[range(feat2_length-1)]
        video_feat2_right = np.zeros_like(video_feat2)
        video_feat2_right[-1] = video_feat2[-1]
        video_feat2_right[range(feat2_length-1)] = video_feat2[range(1,feat2_length)]
        video_feat2_global = np.repeat(video_feat_global,feat2_length,axis=0)

        video_feat1_context = np.concatenate((video_feat1_left,video_feat1,video_feat1_right),axis=1)
        video_feat2_context = np.concatenate((video_feat2_left,video_feat2,video_feat2_right),axis=1)
        # concatenation of all 128 frame feature and 256 frame feature
        # 拼接多尺滑窗 shape = (6,4096)
        
        video_feat=np.concatenate((video_feat1_context,video_feat2_context),axis=0)
        
        #video_feat=np.concatenate((video_feat1,video_feat2),axis=0)


        # 数组转成tensor
        video = torch.Tensor(video_feat)
        sentence = description[index]
        vocab = self.vocab

        # Convert sentence (string) to word ids.
        # 分词
        tokens = nltk.tokenize.word_tokenize(
            str(sentence).lower())
        sentence = []
        # 开头结尾添加分隔符
        sentence.append(vocab('<start>'))
        sentence.extend([vocab(token) for token in tokens])
        sentence.append(vocab('<end>'))
        # 转为tensor
        sentence = torch.Tensor(sentence)
        # video [6,4096] sentence [12] index 2059
        return video, sentence, index

    def __len__(self):
        # 长度按照句子数量，其实就是pair的数量
        return len(self.description)







	
