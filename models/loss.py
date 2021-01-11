import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tools.util import LogSumExp, get_mask, l2norm
from models.CMIL import get_lambda, get_video_score_nms, get_video_score_nms_all,get_video_score_nms_list
import time
import random
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from models.mapping import mapping
from models.IMRAM import frame_by_word


def plot_map(score_maps, map_name):
    score_maps = score_maps.cpu().numpy()

    batch_size = score_maps.shape[0]
    f = plt.figure(figsize=(6,4))
    for i in range(batch_size):
        score_map = score_maps[i]
        plt.matshow(score_map, cmap = plt.cm.cool)
        plt.ylabel("duration")
        plt.xlabel("start time")
        plt.colorbar()
        plt.savefig(os.path.join(str(i)+map_name+'.png'))
        plt.clf()    
    plt.close(f)

def norm(score):
    b,d,t = score.size()
    score = score.view(b,-1)
    score_min = torch.min(score, dim = -1, keepdim = True)[0]
    score_max = torch.max(score, dim = -1, keepdim = True)[0]
    score_norm = (score - score_min)/(score_max-score_min)
    score = score - score_min
    score = score*score_norm
    score = score.view(b,d,t)
    return score

def mask2weight(mask2d, mask_kernel, padding=0):
    # from the feat2d.py,we can know the mask2d is 4-d
    #plot_mask(mask2d.cpu().detach().numpy(),'mask')
    weight = torch.conv2d(mask2d[None,None,:,:].float(),
        mask_kernel, padding=padding)[0, 0]
    #plot_mask(weight.cpu().detach().numpy(),'weight')
    weight[weight > 0] = 1 / weight[weight > 0]
    #plot_mask(weight.cpu().detach().numpy(),'re_weight')
    return weight

def diag(scores, margin):
    b = scores.size(0)
    diagonal = scores.diag().view(scores.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    # compare every diagonal score to scores in its column
    # caption retrieval
    cost_s = (margin + scores - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (margin + scores - d2).clamp(min=0)

    # clear diagonals
    I = torch.eye(scores.size(0)) > .5
    if torch.cuda.is_available():
        I = I.cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)

    # keep the maximum violating negative for each query
    # if self.opt.max_violation:
    #     num = self.opt.neg_num
    #     cost_s = cost_s.sort(1,descending=True)[0][:,:num]
    #     cost_im = cost_im.sort(0,descending=True)[0][:num,:]

    return cost_s.sum()/b + cost_im.sum()/b

class Criterion(nn.Module): # two_stage
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        k = opt.kernel_size
        l = opt.layers
        s = opt.stride

        self.fc = nn.Linear(opt.video_dim, opt.joint_dim)

        self.conv = nn.ModuleList(
            [nn.Conv1d(opt.joint_dim, opt.joint_dim, k, padding=0, stride = s)] # (k - 1) // 2
        )

        for _ in range(l - 1):
            self.conv.append(nn.Conv1d(opt.joint_dim, opt.joint_dim, k, padding=0, stride = s))
        
        self.conv_1d = nn.Conv1d(opt.joint_dim, opt.joint_dim, kernel_size=1)
        self.conv_g1d = nn.Conv1d(opt.joint_dim, opt.joint_dim, kernel_size=k)

    def forward(self, video, words, w_masks, sentences, writer, iters, lam, epoch, iou_map):

        # words[b,l,c]
        video = self.fc(video)
        b = video.size(0)
        postive_map = []
        score_map = []
        g_score_map = []

        for i in range(b):
            word = words[i] # [l,c]
            w_mask = w_masks[i]
            # sentence = sentences[i]

            word = word.unsqueeze(0).repeat(b,1,1) # [l,c] -> [b,l,c]
            w_mask = w_mask.unsqueeze(0).repeat(b,1)
            # sentence = sentence.unsqueeze(0).repeat(b,1)
            
            v_s = frame_by_word(video,None,word,w_mask,self.opt).transpose(1,2)
            v = video.clone().transpose(1,2)

            score = []
            for layer in self.conv:
                v_short = F.max_pool1d(v,self.opt.kernel_size,self.opt.stride)
                v_s_short = F.max_pool1d(v_s,self.opt.kernel_size,self.opt.stride)
                v = layer(v).relu()+v_short
                v_s = layer(v_s).relu()+v_s_short

                v_p = self.conv_1d(v) # [b,c,l]
                v_s_p = self.conv_1d(v_s)
                
                # v = l2norm(v,dim=-1)
                # v_s = l2norm(v_s,dim=-1)
                sim = torch.cosine_similarity(v_p,v_s_p,dim=1) # [b,l,c] -> [b,l]
                score.append(sim) # [b,l]
            
            score = torch.cat(score,dim=1)
            postive_map.append(score[i])
            g_v = l2norm(self.conv_g1d(v).squeeze(),dim=-1)
            # g_score = torch.cosine_similarity(g_v,sentence,dim=-1) # [b,c] -> [b]
            g_s = l2norm(self.conv_g1d(v_s).squeeze(),dim=-1)
            g_score = torch.cosine_similarity(g_v,g_s,dim=1)

            # score_map.append(score.max(dim=1)[0])
            score = get_video_score_nms_list(score,lam,iou_map,i)
            score_map.append(score)
            g_score_map.append(g_score)

        postive_map = torch.stack(postive_map) # [b,l]
        scores = torch.stack(score_map) # [b,b]
        g_scores = torch.stack(g_score_map)

        local_loss = diag(scores,self.opt.global_margin)
        global_loss = diag(g_scores,self.opt.global_margin)

        threshold = self.opt.start_local
        
        if self.training and iters % self.opt.log_step == 0:
            if epoch < threshold:
                writer.add_scalar('global_loss',global_loss,iters)
            else:
                writer.add_scalar('local_loss',local_loss,iters)
            # writer.add_scalar('negative_loss',negative_loss,iters)

        if epoch < threshold:
            loss = global_loss
        else:
            loss = local_loss
        return loss, postive_map

class Criterion_siamese(nn.Module): # 
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        k = opt.kernel_size
        l = opt.layers
        s = opt.stride

        self.fc = nn.Linear(opt.video_dim, opt.joint_dim)

        self.conv = nn.ModuleList(
            [nn.Conv1d(opt.joint_dim, opt.joint_dim, k, padding=0, stride = s)] # (k - 1) // 2
        )

        for _ in range(l - 1):
            self.conv.append(nn.Conv1d(opt.joint_dim, opt.joint_dim, k, padding=0, stride = s))
        
        self.conv_1d = nn.Conv1d(opt.joint_dim, opt.joint_dim, kernel_size=1)

    def forward(self, video, words, w_masks, sentences, writer, iters, lam, iou_map):

        # words[b,l,c]
        video = self.fc(video)
        b = video.size(0)
        postive_map = []
        score_map = []

        for i in range(b):
            word = words[i] # [l,c]
            w_mask = w_masks[i]

            word = word.unsqueeze(0).repeat(b,1,1) # [l,c] -> [b,l,c]
            w_mask = w_mask.unsqueeze(0).repeat(b,1)

            v_s = frame_by_word(video,None,word,w_mask,self.opt).transpose(1,2)
            v = video.clone().transpose(1,2)

            score = []
            for layer in self.conv:
                v_short = F.max_pool1d(v,self.opt.kernel_size,self.opt.stride)
                v_s_short = F.max_pool1d(v_s,self.opt.kernel_size,self.opt.stride)
                v = layer(v).relu()+v_short
                v_s = layer(v_s).relu()+v_s_short
                v_p = self.conv_1d(v)
                v_s_p = self.conv_1d(v_s) # [b,c,l]
                sim = torch.cosine_similarity(v_p,v_s_p,dim=1) # [b,l]
                score.append(sim) # [b,l]

            score = torch.cat(score,dim=1)
            postive_map.append(score[i])
            score_map.append(score.max(dim=1)[0])

        postive_map = torch.stack(postive_map) # [b,l]
        scores = torch.stack(score_map) # [b,b]
        loss = diag(scores,self.opt.global_margin)
        
        if self.training and iters % self.opt.log_step == 0:
            writer.add_scalar('loss',loss,iters)

        return loss, postive_map

class Criterion_cosine(nn.Module): # cosine
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        k = opt.kernel_size
        l = opt.layers
        s = opt.stride

        self.fc = nn.Linear(opt.video_dim,opt.joint_dim)

        self.conv = nn.ModuleList(
            [nn.Conv1d(opt.joint_dim, opt.joint_dim, k, padding=0, stride = s)] # (k - 1) // 2
        )

        for _ in range(l - 1):
            self.conv.append(nn.Conv1d(opt.joint_dim, opt.joint_dim, k, padding=0, stride = s))
        
        self.conv_1d = nn.Conv1d(opt.joint_dim, opt.joint_dim, kernel_size=1)

    def forward(self, video, words, w_masks, sentences, writer, iters, lam, iou_map):

        # words[b,l,c]
        video = self.fc(video)
        video = video.transpose(1,2) #[b,l,c] -> [b,c,l]
        b = video.size(0)
        postive_map = []
        score_map = []
        # smooth = torch.tensor(0.).cuda()

        for i in range(b):
            word = words[i] # [l,c]
            w_mask = w_masks[i]
            word = word.unsqueeze(0).repeat(b,1,1) # [l,c] -> [b,l,c]
            w_mask = w_mask.unsqueeze(0).repeat(b,1)
            score = []
            v_tmp = video.clone()
            cnt = 0
            for layer in self.conv:
                # v_short = F.max_pool1d(v_tmp,self.opt.kernel_size,self.opt.stride)
                v_tmp = layer(v_tmp).relu()#+v_short
                if cnt >= self.opt.start_layer - 1:
                    v = self.conv_1d(v_tmp) # [b,c,l]
                    v = v.transpose(1,2) # [b,c,l] -> [b,l,c]
                    v_s = frame_by_word(v,None,word,w_mask,self.opt)
                    v = l2norm(v,dim=-1)
                    v_s = l2norm(v_s,dim=-1)
                    sim = torch.cosine_similarity(v,v_s,dim=-1) # [b,l,c] -> [b,l]

                    # tmp = sim[i]
                    # for j in range(len(tmp)-1):
                    #     smooth += (tmp[j+1] - tmp[j])**2
                    score.append(sim) # [b,l]
                cnt += 1
            
            score = torch.cat(score,dim=1)
            postive_map.append(score[i])
            
            # length = score.size(1)
            # length = max(int(length*(1-lam)),1)
            # score = score.sort(dim=1,descending=True)[0][:,:length]
            # # score_map.append(score.max(dim=1)[0])
            # score_map.append(score.mean(dim=1))
            score = get_video_score_nms_list(score,lam,iou_map,i)
            score_map.append(score)
        postive_map = torch.stack(postive_map) # [b,l]
        scores = torch.stack(score_map) # [b,b]

        # sparse_loss = F.relu(postive_map).sum()/b

        loss = diag(scores,self.opt.global_margin)

        # smooth = smooth/b
        if self.training and iters % self.opt.log_step == 0:
            writer.add_scalar('loss',loss,iters)
            # writer.add_scalar('smooth_loss',smooth,iters)
            # writer.add_scalar('sparse_loss',sparse_loss,iters)
    
        return loss, postive_map #+smooth*self.opt.smooth_lam +sparse_loss*self.opt.smooth_lam


class Criterion1(nn.Module): #local
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        k = opt.kernel_size
        l = opt.layers
        s = opt.stride

        self.norm = nn.BatchNorm1d(opt.video_dim,affine=False)
        self.norm_v = nn.BatchNorm1d(opt.joint_dim,affine=False)
        self.norm_w = nn.BatchNorm1d(opt.joint_dim,affine=False)
        self.conv = nn.ModuleList(
            [nn.Conv1d(opt.video_dim, opt.joint_dim, k, padding=0, stride = s)] # (k - 1) // 2
        )

        for _ in range(l - 1):
            self.conv.append(nn.Conv1d(opt.joint_dim, opt.joint_dim, k, padding=0, stride = s))
        
        self.conv_1d = nn.Conv1d(opt.joint_dim, opt.joint_dim, kernel_size=1)

    def forward(self, video, words, w_masks, sentences, writer, iters, lam, iou_map):

        # words[b,l,c]
        
        video = video.transpose(1,2) #[b,l,c] -> [b,c,l]
        video = self.norm(video)
        b = video.size(0)
        postive_map = []
        score_map = []
        v = []

        for layer in self.conv:
            video = layer(video).relu()
            v.append(video)
        v = torch.cat(v,dim=-1)

        v = self.conv_1d(v) # [b,c,l]
        v = self.norm_v(v) 
        words = words.transpose(1,2) # [b,l,c] -> [b,c,l]
        words = self.norm_w(words)

        v = v.transpose(1,2) # [b,c,l] -> [b,l,c]
        words = words.transpose(1,2)
        # v = l2norm(v,dim=-1)
        # v_s = l2norm(v_s,dim=-1)

        for i in range(b):
            word = words[i] # [l,c]
            w_mask = w_masks[i]
            word = word.unsqueeze(0).repeat(b,1,1) # [l,c] -> [b,l,c]
            w_mask = w_mask.unsqueeze(0).repeat(b,1)
            v_s = frame_by_word(v,None,word,w_mask,self.opt)
            score = torch.cosine_similarity(v,v_s,dim=-1) # [b,l,c] -> [b,l]

            postive_map.append(score[i])
            score_map.append(score.max(dim=1)[0])
            # score_map.append(score.mean(dim=1))
            # score = get_video_score_nms_list(score,lam,iou_map,i)
            # score_map.append(score)
        postive_map = torch.stack(postive_map) # [b,l]
        scores = torch.stack(score_map) # [b,b]

        loss = diag(scores,self.opt.global_margin)

        if self.training and iters % self.opt.log_step == 0:
            writer.add_scalar('loss',loss,iters)
    
        return loss, postive_map
