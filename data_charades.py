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
    Load precomputed captions and image features
    """

    def __init__(self, data_split, dpath, vocab):
        """
        根据选定的数据集，读取csv文件，获取视频列表和标注信息，例如charades_test.csv
        """
        self.vocab=vocab
        path=dpath+"/Caption/charades_"+ str(data_split) + ".csv"
        df=pandas.read_csv(path)
        df_temp = pandas.read_csv(path,dtype={'ID': object})
        # 视频名称列表
        self.inds = df_temp['video']
        # 描述语句列表
        self.desc=df['description']
        # 数据集划分
        self.data_split=data_split
        # 数据集地址
        self.data_path=dpath

    def __getitem__(self, index):

        img_id = index
        inds=self.inds
        desc=self.desc
        # load C3D feature 单个视频读取
        video_feat_file=self.data_path+"/c3d_features/"+str(inds[index])+".mat"
        video_feat_mat = sio.loadmat(video_feat_file)
        video_feat=video_feat_mat['feature']
        # 读取出来就是128的滑窗
        #data_path = '/home/share/wangyunxiao/Charades/CHARADES_C3D/concate_16'
        #video_feat = np.load(os.path.join(data_path,str(inds[index]+'.npy')))
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

        video_feat1_context = np.concatenate((video_feat1_left,video_feat1,video_feat1_right,video_feat1_global),axis=1)
        video_feat2_context = np.concatenate((video_feat2_left,video_feat2,video_feat2_right,video_feat2_global),axis=1)
        # concatenation of all 128 frame feature and 256 frame feature
        # 拼接多尺滑窗 shape = (6,4096)
        
        video_feat=np.concatenate((video_feat1_context,video_feat2_context),axis=0)
        
        #video_feat=np.concatenate((video_feat1,video_feat2),axis=0)


        # 数组转成tensor
        image = torch.Tensor(video_feat)
        caption = desc[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        # 分词
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())
        caption = []
        # 开头结尾添加分隔符
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        # 转为tensor
        target = torch.Tensor(caption)
        # image [6,4096] target [12] index 2059 img_id 2059
        return image, target, index, img_id

    def __len__(self):
        # 长度按照句子数量，其实就是pair的数量
        return len(self.desc)

def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
       貌似主要是对视频和文本进行了zero padding，然后合并成一个列表，也就是batch
    Args:
        data: list of (image, caption) tuple.

    Returns:
        images: torch tensor of shape (batch_size, feature_size).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    # lengths_img是滑动窗口提取的特征数目
    lengths_img = [len(im) for im in images]
    # 貌似就进行了一个zero padding，补足滑窗的数目，每个滑窗的特征是4096维
    target_images = torch.zeros(len(images), max(lengths_img), 4096*4) 
    for i, im in enumerate(images):
        end = lengths_img[i]
        target_images[i, :end,] = im[:end,]

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    # 还是zero padding
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    # 依次是一个batch的padding后的视频、padding后的文本，文本的单词数目，视频的滑窗数目，pair的序号
    return target_images, targets, lengths, lengths_img, ids



def get_charades_loader(data_split, dpath, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # 封装了一个dataset对象
    dset = Charades(data_split, dpath, vocab)
    # 封装了一个dataloader对象
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader



def get_loaders(data_name, vocab, crop_size, batch_size, workers, opt):
    """对get_charades_loader的再封装，返回train_loader,val_loader"""
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_charades_loader('train', dpath, vocab, opt,
                                          batch_size, True, workers)
    val_loader = get_charades_loader('val', dpath, vocab, opt,
                                        batch_size, False, workers)


    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, crop_size, batch_size,
                    workers, opt):
    """对get_charades_loader的再封装，返回test_loader"""
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_charades_loader(split_name, dpath, vocab, opt,
                                         batch_size, False, workers)
										 

    return test_loader

	
