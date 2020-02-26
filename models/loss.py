import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tools.util import cosine_sim
from torch.nn.utils.rnn import pack_padded_sequence

def video_score1(score_map):
    num = score_map.size(0)
    score_map = 1-score_map.view(num,-1)
    print("score_map:",score_map.shape)
    print("score_map:",score_map)
    print("score_prod:",torch.prod(score_map,dim=-1))
    print("score_prod:",torch.prod(score_map,dim=-1).shape)
    exit()
    return 1 - torch.prod(score_map,dim=-1)

def video_score2(score_map):
    num = score_map.size(0)
    score_map = score_map.view(num,-1)
    #print("score_map:",score_map.shape)
    #print("score_map:",score_map)
    score = torch.max(score_map,dim=-1)[0]
    #print("score:",score)
    #print("score:",score.shape)
    return score
def video_score(score_map):
    r = 1
    score = torch.log(torch.sum(torch.exp(score_map*r),dim=-1))/r
    return score
def Criterion(score,batch_size,negative_num,opt):
    """
    Compute loss
    video [b,c,d,t] sentence [b,c]
    """
    # score [b,d*t]
    '''
    score_max = torch.max(score,dim=-1)[0]
    
    positive_score = score_max[:batch_size]
    negative_score = score_max[batch_size:]
    '''
    score = video_score(score)
    positive_score = score[:batch_size]
    negative_score = score[batch_size:]

    negative_score = negative_score.view(-1,negative_num)
    positive_score = positive_score.unsqueeze(1).expand_as(negative_score)
    global_loss = (opt.global_margin + negative_score - positive_score).clamp(min=0)
    global_loss = global_loss.sum()#/batch_size
    '''
    local_loss = torch.zeros(1).cuda()
    for i in range(batch_size):
        # 展平
        temp = positive_map[i].view(-1)
        # 降序排列
        temp = torch.sort(temp,descending=True)[0]
        num = torch.sum(temp > 0.01)
        #pos = int(np.ceil(num*0.1))
        pos = torch.ceil(num.float()*0.1).int()
        local_loss += (opt.local_margin + torch.mean(temp[num-pos:num]) - torch.mean(temp[:pos])).clamp(min=0)

    local_loss = local_loss/batch_size
    '''
    return global_loss#+local_loss
"""
def Criterion1(positive_map, negative_map,opt):
    mask = get_mask(opt.temporal_scale).unsqueeze(0)

    batch_size= positive_map.size(0)
    mask = mask.repeat(batch_size,1,1)
    positive_map = positive_map*mask

    mask = mask.repeat(opt.negative_num,1,1)
    negative_map = negative_map*mask

    positive_score = video_score(positive_map)
    negative_score = video_score(negative_map)
    label = torch.cat((torch.ones(batch_size),torch.zeros(batch_size*opt.negative_num))).cuda()
    #score = torch.cat((positive_score,negative_score))>0.5
    #score = score.float()
    print('score:',torch.cat((positive_score,negative_score)))
    print('label:',label)
    global_loss = F.binary_cross_entropy(torch.cat((positive_score,negative_score)),label)
    '''
    negative_score = negative_score.view(-1,opt.negative_num)
    positive_score = positive_score.unsqueeze(1).expand_as(negative_score)
    global_loss = (opt.global_margin + negative_score - positive_score).clamp(min=0)
    global_loss = global_loss.sum()#/batch_size
    '''
    local_loss = torch.zeros(1).cuda()
    for i in range(batch_size):
        # 展平
        temp = positive_map[i].view(-1)
        # 降序排列
        temp = torch.sort(temp,descending=True)[0]
        num = torch.sum(temp > 0.01)
        #pos = int(np.ceil(num*0.1))
        pos = torch.ceil(num.float()*0.1).int()
        local_loss += (opt.local_margin + torch.mean(temp[num-pos:num]) - torch.mean(temp[:pos])).clamp(min=0)

    local_loss = local_loss/batch_size
    return global_loss#+local_loss
"""