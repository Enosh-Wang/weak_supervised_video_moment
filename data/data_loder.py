import torch
from data.Charades import Charades
from data.ActivityNet import ActivityNet
from torch.utils.data import DataLoader
import os

def collate_fn_vocab(data):
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
    data.sort(key=lambda x: len(x[1]), reverse=True)
    videos, sentences, index = zip(*data)

    # Merge videos (convert tuple of 3D tensor to 4D tensor)
    # lengths_img是滑动窗口提取的特征数目
    video_lengths = [len(video) for video in videos]
    # 貌似就进行了一个zero padding，补足滑窗的数目，每个滑窗的特征是4096维
    video_padded = torch.zeros(len(videos), max(video_lengths), videos[0].size(1)) 
    for i, video in enumerate(videos):
        end = video_lengths[i]
        video_padded[i, :end] = video[:end]

    # Merge sentences (convert tuple of 1D tensor to 2D tensor)
    # 还是zero padding
    sentence_lengths = [len(sentence) for sentence in sentences]
    sentence_padded = torch.zeros(len(sentences), max(sentence_lengths)).long()
    for i, sentence in enumerate(sentences):
        end = sentence_lengths[i]
        sentence_padded[i, :end] = sentence[:end]
    # 依次是一个batch的padding后的视频、padding后的文本，文本的单词数目，视频的滑窗数目，pair的序号
    return video_padded, sentence_padded, sentence_lengths, video_lengths, index

def get_data_loader_vocab(opt, vocab, data_split, shuffle=True):
    """Returns torch.utils.data.DataLoader for  dataset."""
    # 封装了一个dataset对象
    data_path = os.path.join(opt.data_path, opt.dataset)
    if opt.dataset ==  'ActivityNet':
        data_loader = DataLoader(dataset=ActivityNet(data_split, data_path, vocab),
                                batch_size=opt.batch_size,
                                shuffle=shuffle,
                                pin_memory=True,
                                collate_fn=collate_fn)
    elif opt.dataset == 'Charades':
        data_loader = DataLoader(dataset=Charades(data_split, data_path, vocab),
                                batch_size=opt.batch_size,
                                shuffle=shuffle,
                                pin_memory=True,
                                collate_fn=collate_fn)
    return data_loader

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
    data.sort(key=lambda x: len(x[1]), reverse=True)
    videos, sentences, index = zip(*data)

    # Merge videos (convert tuple of 3D tensor to 4D tensor)
    # lengths_img是滑动窗口提取的特征数目
    video_lengths = [len(video) for video in videos]
    # 貌似就进行了一个zero padding，补足滑窗的数目，每个滑窗的特征是4096维
    video_padded = torch.zeros(len(videos), max(video_lengths), videos[0].size(1)) 
    for i, video in enumerate(videos):
        end = video_lengths[i]
        video_padded[i, :end] = video[:end]

    # Merge sentences (convert tuple of 1D tensor to 2D tensor)
    # 还是zero padding
    sentence_lengths = [len(sentence) for sentence in sentences]
    sentence_padded = torch.zeros(len(sentences), max(sentence_lengths),300)#.long()
    for i, sentence in enumerate(sentences):
        end = sentence_lengths[i]
        sentence_padded[i, :end] = sentence[:end]
    # 依次是一个batch的padding后的视频、padding后的文本，文本的单词数目，视频的滑窗数目，pair的序号
    return video_padded, sentence_padded, sentence_lengths, video_lengths, index