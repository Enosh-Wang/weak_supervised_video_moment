import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import math

def multihead_mask(x, lengths):
    seq_size, batch_size, _ = x.size()
    key_padding_mask = torch.BoolTensor(batch_size,seq_size)
    for i in range(batch_size):
        key_padding_mask[i,:lengths[i]] = False
        key_padding_mask[i,lengths[i]:] = True
    key_padding_mask = key_padding_mask.cuda()
    return key_padding_mask

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1).sqrt()
    X = X / norm[:,None]
    return X


def cosine_similarity(x1, x2):
    """Returns cosine similarity based attention between x1 and x2, computed along dim."""
    # 原始代码
    # batch 矩阵相乘 x1[128,26,1024] x2[128,1024] w1[128,26,1]
    # w1=torch.bmm(x1, x2.unsqueeze(2))
    # <!-改为MIL

    batch_size = x1.size()[0]
    x2 = x2.repeat(batch_size,1,1) # [128,128,1024]
    x1 = x1.permute(0,2,1) # [128,1024,14]
    w1 = torch.bmm(x2,x1) # [128,128,14]

    #x2 = x2.permute(0,2,1)
    #w1 = torch.bmm(x1,x2)
    return w1

class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    # 标准的矩阵乘法 结果[滑窗个数，单词个数]
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).squeeze(2).sqrt().t()
    return score

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.add_scalar(prefix + k, v.val, global_step=step)

def cIoU(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
    return float(intersection)/union

def t2i( df, attn_index, video_length, is_training, npts=None):
    """
    Text->videos (Image Search)
    videos: (N, K) matrix of videos
    Captions: (N, K) matrix of captions
    """
    # 读取GT
    start_segment=df['start_segment']
    end_segment=df['end_segment']
	
    if npts is None:
        # pair的数目？
        npts = attn_index.shape[0]

    correct_num05=0
    correct_num07=0
    correct_num03=0

    R5IOU5=0
    R5IOU7=0
    R5IOU3=0
    R10IOU3=0
    R10IOU5=0
    R10IOU7=0
	
    for index in range(int(npts)):
        # index应该是指pair的索引
        # 匹配结果的排序列表
        att_inds=attn_index[index,:]
        # 视频的长度
        len_img=video_length[index]
        # 读取GT
        gt_start=start_segment[index]
        gt_end=end_segment[index]
        # 因为将128和256的滑窗拼接在了一起，所以前2/3的特征是128滑窗的，后1/3的特征是256滑窗的
        break_128=np.floor(len_img*2/3)
        # 起始帧
        rank1_start=att_inds[0]
        if (rank1_start<break_128):
           # 128的滑窗
           rank1_start_seg =rank1_start*128
           rank1_end_seg = rank1_start_seg+128
        else:
           # 256的滑窗
           rank1_start_seg =(rank1_start-break_128)*256
           rank1_end_seg = rank1_start_seg+256
        
        # 计算IOU
        iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
        if iou>=0.5:
           correct_num05+=1
        if iou>=0.7:
           correct_num07+=1
        if iou>=0.3:
           correct_num03+=1

        for j1 in range(5):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.5:
               R5IOU5+=1
               break
			   
        for j1 in range(5):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.7:
               R5IOU7+=1
               break
			   
        for j1 in range(5):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.3:
               R5IOU3+=1
               break
			   
        for j1 in range(10):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.5:
               R10IOU5+=1
               break
			   
        for j1 in range(10):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.7:
               R10IOU7+=1
               break
			   
        for j1 in range(10):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.3:
               R10IOU3+=1
               break    

	############################

    # Compute metrics
    R1IoU05=correct_num05
    R1IoU07=correct_num07
    R1IoU03=correct_num03
    total_length=attn_index.shape[0]
    #print('total length',total_length)
    if is_training == False:
        print("R@1 IoU0.3: %f" %(R1IoU03/float(total_length)))
        print("R@5 IoU0.3: %f" %(R5IOU3/float(total_length)))
        print("R@10 IoU0.3: %f" %(R10IOU3/float(total_length)))
        
        print("R@1 IoU0.5: %f" %(R1IoU05/float(total_length)))
        print("R@5 IoU0.5: %f" %(R5IOU5/float(total_length)))
        print("R@10 IoU0.5: %f" %(R10IOU5/float(total_length)))
        
        print("R@1 IoU0.7: %f" %(R1IoU07/float(total_length)))
        print("R@5 IoU0.7: %f" %(R5IOU7/float(total_length)))
        print("R@10 IoU0.7: %f" %(R10IOU7/float(total_length)))
	
	
    return R1IoU03, R1IoU05, R1IoU07
