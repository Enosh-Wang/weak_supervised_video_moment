import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.util import order_sim,cosine_sim
from torch.nn.utils.rnn import pack_padded_sequence

def Criterion(similarity, captions, caption_lengths, video_lengths, caption_predict,margin,max_violation=False):
    """
    Compute loss
    """
    #target_captions = pack_padded_sequence(captions,caption_lengths)[0]
    #caption_loss = F.cross_entropy(caption_predict,target_captions)
    # 取对角线元素 2D -> 1D
    # view的作用类似reshape
    scores = torch.max(similarity,2)[0]
    diagonal = scores.diag().view(scores.size(0), 1)
    # 扩展成与scores相同的大小
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)
    
    # compare every diagonal score to scores in its column
    # caption retrieval

    cost_s = (margin + scores - d1).clamp(min=0)

    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (margin + scores - d2).clamp(min=0)

    # clear diagonals
    # eye:返回一个2维张量，对角线位置全1，其它位置全0
    mask = torch.eye(scores.size(0)) > .5
    I = mask
    if torch.cuda.is_available():
        I = I.cuda()
    # masked_fill_:在mask值为1的位置处用value填充。mask的元素个数需和本tensor相同，但尺寸可以不同。
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)
    # keep the maximum violating negative for each query
    # 用max代替sum
    if max_violation:
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]
    
    
    size = similarity.size()
    cost_p = torch.zeros((size[0],size[2])).cuda()
    cost_n = torch.zeros((size[0],size[0])).cuda()
    for i in range(size[0]):
        
        positive_bag = similarity[i,i,:]
        tp_tensor,tp_arg = torch.max(positive_bag,dim=0,keepdim=True)
        tp1 = tp_tensor.expand_as(similarity[i,i,:])
        cost_p[i] = (margin + positive_bag - tp1).clamp(min=0)
        #cost_p[i] = (margin - positive_bag + tp1).clamp(min=0)
        cost_p[i,tp_arg] = 0

        negative_bag = similarity[:,i,tp_arg]
        negative_bag = torch.squeeze(negative_bag)
        tp2 = tp_tensor.expand_as(negative_bag)
        cost_n[i] = (margin + negative_bag - tp2).clamp(min=0)
        #cost_n[i] = (margin - negative_bag + tp2).clamp(min=0)
        cost_n[i,i] = 0
        
    #cost_p = cost_p.max(1)[0]
    #cost_n = cost_n.max(1)[0]

    '''
    # 时间平滑项
    lambda_1 = 8e-5
    size = similarity.size()
    cost_smooth = Variable(torch.zeros(size[0],size[2]),requires_grad = True)
    cost_smooth = cost_smooth.cuda()
    for i in range(size[0]):
        a = range(video_lengths[i]-1)
        b = range(1,video_lengths[i])
        cost_smooth[i,:video_lengths[i]-1] = similarity[i,i,a]-similarity[i,i,b]
    # 时间稀疏项
    lambda_2 = 8e-5
    cost_sparse = Variable(torch.zeros(size[0],size[2]),requires_grad = True)
    cost_sparse = cost_sparse.cuda()
    for i in range(size[0]):
        cost_sparse[i,:video_lengths[i]] = similarity[i,i,:video_lengths[i]]
    '''
    return cost_p.sum()+cost_n.sum()#+caption_loss#+cost_n.sum()#cost_im.sum()+cost_s.sum() #+  #+ lambda_1*cost_smooth.pow(2).sum() + lambda_2*cost_sparse.abs().sum()
