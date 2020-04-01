import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import math
import pickle

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def LogSumExp(X, lambda_lse, dim):
    score = torch.log(torch.sum(torch.exp(X*lambda_lse),dim))/lambda_lse
    return score

def get_mask(tscale):

    bm_mask = []
    # 遍历每行，逐行计算掩码，逐行在末尾增加0
    for idx in range(tscale):
        mask_vector = [1 for i in range(tscale - idx)
                    ] + [0 for i in range(idx)]
        bm_mask.append(mask_vector)
    bm_mask = np.array(bm_mask, dtype=np.int)
    return torch.IntTensor(bm_mask).cuda()

def get_mask_spare(tscale):
         
    bm_mask = []
    # 遍历每行，逐行计算掩码，逐行在末尾增加0
    for duration in range(1, tscale+1):
        mask_vector = []
        k = np.ceil(np.log2(duration/6))
        s = 2**(k-1)
        if k == 1:
            s2 = 0
        else:
            s2 = 2**(k+2)-1

        for start in range(tscale-duration+1):

            if np.mod(start,s) == 0 and np.mod(start+duration-s2,s) == 0:
                mask_vector.append(1)
            else:
                mask_vector.append(0)
                
        mask_vector += [0 for i in range(duration-1)]
        bm_mask.append(mask_vector)
    bm_mask = np.array(bm_mask, dtype=np.int)
    return torch.IntTensor(bm_mask).cuda()

def load_glove(glove_path):
    with open(glove_path, 'rb') as f:
        word2vec = pickle.load(f)
    return word2vec

def multihead_mask(x, lengths):
    seq_size, batch_size, _ = x.size()
    key_padding_mask = torch.BoolTensor(batch_size,seq_size)
    for i in range(batch_size):
        key_padding_mask[i,:lengths[i]] = False
        key_padding_mask[i,lengths[i]:] = True
    key_padding_mask = key_padding_mask.cuda()
    return key_padding_mask

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

    def __init__(self, d_model, dropout=0., max_len=5000):
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

def t2i( df, filename, is_training):
    """
    Text->videos (Image Search)
    videos: (N, K) matrix of videos
    Captions: (N, K) matrix of captions
    """
    # 读取GT
    start_time_list = df['start_time']
    end_time_list = df['end_time']
    duration_list = df['duration']
	
    with open(filename,'rb') as f:
        top10 = pickle.load(f)

    total_length = len(df)

    correct_num05=0
    correct_num07=0
    correct_num03=0

    R5IOU5=0
    R5IOU7=0
    R5IOU3=0
    R10IOU3=0
    R10IOU5=0
    R10IOU7=0
	
    for index in range(total_length):
        # 读取GT
        gt_start=start_time_list[index]
        gt_end=end_time_list[index]
        duration = duration_list[index]
        proposal = top10[index]
        start = proposal[:,0]*duration
        end = proposal[:,1]*duration
        num = len(start)
        
        # 计算IOU
        # r@1
        iou = cIoU((gt_start, gt_end),(start[0], end[0]))
        if iou>=0.7:
           correct_num07+=1
        if iou>=0.5:
           correct_num05+=1
        if iou>=0.3:
           correct_num03+=1
        # R@5
        for j in range(min(5,num)):
            iou = cIoU((gt_start, gt_end),(start[j], end[j]))
            if iou>=0.7:
               R5IOU7+=1
               break
        for j in range(min(5,num)):
            iou = cIoU((gt_start, gt_end),(start[j], end[j]))
            if iou>=0.5:
               R5IOU5+=1
               break
        for j in range(min(5,num)):
            iou = cIoU((gt_start, gt_end),(start[j], end[j]))
            if iou>=0.3:
               R5IOU3+=1
               break
		# R@10
        for j in range(min(10,num)):
            iou = cIoU((gt_start, gt_end),(start[j], end[j]))
            if iou>=0.7:
               R10IOU7+=1
               break
        for j in range(min(10,num)):			   
            iou = cIoU((gt_start, gt_end),(start[j], end[j]))
            if iou>=0.5:
               R10IOU5+=1
               break
        for j in range(min(10,num)):
            iou = cIoU((gt_start, gt_end),(start[j], end[j]))
            if iou>=0.3:
               R10IOU3+=1
               break    

	############################

    # Compute metrics
    R1IOU05=correct_num05
    R1IOU07=correct_num07
    R1IOU03=correct_num03
    if is_training == False:
        print("R@1 IoU0.3: %f" %(R1IOU03/float(total_length)))
        print("R@5 IoU0.3: %f" %(R5IOU3/float(total_length)))
        print("R@10 IoU0.3: %f" %(R10IOU3/float(total_length)))
        
        print("R@1 IoU0.5: %f" %(R1IOU05/float(total_length)))
        print("R@5 IoU0.5: %f" %(R5IOU5/float(total_length)))
        print("R@10 IoU0.5: %f" %(R10IOU5/float(total_length)))
        
        print("R@1 IoU0.7: %f" %(R1IOU07/float(total_length)))
        print("R@5 IoU0.7: %f" %(R5IOU7/float(total_length)))
        print("R@10 IoU0.7: %f" %(R10IOU7/float(total_length)))
	
	
    return R1IOU03/float(total_length), R1IOU05/float(total_length), R1IOU07/float(total_length)
