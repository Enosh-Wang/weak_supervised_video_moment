import torch
from data.Charades import Charades,CharadesGCN
from data.ActivityNet import ActivityNet,ActivityNetGCN
from torch.utils.data import DataLoader
import os
import numpy as np

def get_data_loader(opt, word2vec, data_split, shuffle=True):
    """Returns torch.utils.data.DataLoader for  dataset."""
    # 封装了一个dataset对象
    data_path = os.path.join(opt.data_path, opt.dataset)
    if opt.dataset ==  'ActivityNet':
        data_loader = DataLoader(dataset=ActivityNet(data_split, data_path, word2vec),
                                batch_size=opt.batch_size,
                                shuffle=shuffle,
                                pin_memory=True,
                                collate_fn=collate_fn)
    elif opt.dataset == 'Charades':
        data_loader = DataLoader(dataset=Charades(data_split, data_path, word2vec),
                                batch_size=opt.batch_size,
                                shuffle=shuffle,
                                pin_memory=True,
                                collate_fn=collate_fn)
    return data_loader

def collate_fn(data):
    """Build mini-batch tensors from a list of (video, sentence) tuples.
       貌似主要是对视频和文本进行了zero padding，然后合并成一个列表，也就是batch
    Args:
        data: list of (video, sentence) tuple.

    Returns:
        videos: torch tensor of shape (batch_size, feature_size).
        sentence: torch tensor of shape (batch_size, padded_length).
        sentence_lengths: list; valid length for each padded sentence.
    """
    # Sort a data list by sentence length
    # 根据文本的长度对数据排序
    data.sort(key=lambda x: len(x[1]), reverse=True)
    videos, sentences, index, video_name = zip(*data)
    # Merge videos (convert tuple of 3D tensor to 4D tensor)
    video_lengths = [len(video) for video in videos]
    video_padded = torch.zeros(len(videos), max(video_lengths), videos[0].size(1)) 
    for i, video in enumerate(videos):
        video_padded[i] = video
    # Merge sentences (convert tuple of 1D tensor to 2D tensor)
    # 还是zero padding
    sentence_lengths = [len(sentence) for sentence in sentences]
    sentence_padded = torch.zeros(len(sentences), max(sentence_lengths),300)
    for i, sentence in enumerate(sentences):
        end = sentence_lengths[i]
        sentence_padded[i, :end] = sentence[:end]
    # 依次是一个batch的视频、padding后的文本，文本的单词数目，视频的滑窗数目，pair的序号
    return video_padded, sentence_padded, sentence_lengths, index ,video_name

def get_data_loader_GCN(opt, word2vec, data_split, shuffle=True):
    """Returns torch.utils.data.DataLoader for  dataset."""
    # 封装了一个dataset对象
    data_path = os.path.join(opt.data_path, opt.dataset)
    if opt.dataset ==  'ActivityNet':
        data_loader = DataLoader(dataset=ActivityNetGCN(data_split, data_path, word2vec),
                                batch_size=opt.batch_size,
                                shuffle=shuffle,
                                pin_memory=True,
                                collate_fn=collate_fn_GCN)
    elif opt.dataset == 'Charades':
        data_loader = DataLoader(dataset=CharadesGCN(data_split, data_path, word2vec),
                                batch_size=opt.batch_size,
                                shuffle=shuffle,
                                pin_memory=True,
                                collate_fn=collate_fn_GCN)
    return data_loader

def collate_fn_GCN(data):

    # Sort a data list by sentence length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    videos, video_mat, sentences, id2pos, sentence_mat, index, video_name = zip(*data)
    # Merge videos (convert tuple of 3D tensor to 4D tensor)
    video_lengths = [len(video) for video in videos]
    video = torch.stack(videos, dim=0)
    video_mat = torch.stack(video_mat, dim=0)

    # 还是zero padding
    sentence_lengths = [len(sentence) for sentence in sentences]
    batch_size = len(sentences)
    length = max(sentence_lengths)

    sentence_padded = torch.zeros(batch_size,length,300)
    id2pos_padded = np.zeros(batch_size,length)
    mat_padded = np.zeros(batch_size,length,length)
    
    for i in range(batch_size):
        end = sentence_lengths[i]
        sentence_padded[i,:end] = sentences[i]
        # 对分词序号padding
        id2pos_padded[i,:end] = id2pos[i]
        # 对依赖矩阵padding
        mat_padded[i,:end,:end] = sentence_mat[i]
    # 依次是一个batch的视频、padding后的文本，文本的单词数目，视频的滑窗数目，pair的序号
    return video, video_mat, sentence_padded, id2pos_padded, mat_padded, sentence_lengths, video_lengths, index ,video_name