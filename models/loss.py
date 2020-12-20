import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tools.util import LogSumExp, get_mask
from models.CMIL import get_lambda, get_video_score_nms, get_video_score_nms_all
import time
import random
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from models.mapping import mapping

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
    
class Criterion(nn.Module):
    def __init__(self, opt, v_mask):
        super().__init__()
        self.opt = opt
        k = opt.kernel_size
        l = opt.map_layers

        mask_kernel = torch.ones(1,1,k,k).cuda()
        first_padding = (k - 1) * l // 2

        self.weights = [
            mask2weight(v_mask, mask_kernel, padding=first_padding) 
        ]
        self.conv_map = nn.ModuleList(
            [mapping(opt, opt.joint_dim, opt.joint_dim, k, padding=first_padding)]
        )

        for _ in range(l - 1):
            self.weights.append(mask2weight(self.weights[-1] > 0, mask_kernel))
            self.conv_map.append(mapping(opt, opt.joint_dim, opt.joint_dim, k))
        
        self.conv_2d = nn.Conv2d(opt.joint_dim,1,kernel_size=1)

        
    def forward(self, v_map, words, w_masks, writer, iters, lam, v_mask, valid_num, iou_maps):

        # v_map[b,c,d,t] words[b,l,c]
        b,c,d,t = v_map.size()
        postive_map = []
        score_map = []
        negative_loss = []

        for i in range(b):
            word = words[i] # [l,c]
            w_mask = w_masks[i]
            word = word.unsqueeze(0).repeat(b,1,1) # [l,c] -> [b,l,c]
            w_mask = w_mask.unsqueeze(0).repeat(b,1)
            temp = v_map.clone()
            for layer, weight in zip(self.conv_map, self.weights):
                temp = layer(temp,v_mask, word, w_mask) #* weight
            
            score = self.conv_2d(temp).sigmoid().squeeze(1) # [b,d,t]

            score = score.masked_fill(v_mask == 0, float('-inf'))
            postive_map.append(score[i]) #[d,t]
            # plot_map(score,'tacos')
            # exit()
            score = score.view(b,-1)
            score,neg_score = get_video_score_nms(score, valid_num, lam, iou_maps, i)

            negative_loss.append(neg_score)
            score_map.append(score)

        negative_loss = torch.stack(negative_loss).mean()
        postive_map = torch.stack(postive_map) # [b,d,t]
        scores = torch.stack(score_map) # [b,b]

        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.opt.global_margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.opt.global_margin + scores - d2).clamp(min=0)

        # clear diagonals
        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.opt.max_violation:
            num = self.opt.neg_num
            cost_s = cost_s.sort(1,descending=True)[0][:,:num]
            cost_im = cost_im.sort(0,descending=True)[0][:num,:]
        
        if self.training and iters % self.opt.log_step == 0:
            writer.add_scalar('cost_s',cost_s.sum()/b,iters)
            writer.add_scalar('cost_im',cost_im.sum()/b,iters)
            writer.add_scalar('negative_loss',negative_loss,iters)

        return cost_s.sum()/b + cost_im.sum()/b + negative_loss, postive_map


def Caption_Criterion(pred, word_id, sentence_lengths):
    word_id = word_id.transpose(0,1).cuda()
    target_captions = pack_padded_sequence(word_id,sentence_lengths)[0]
    caption_loss = F.cross_entropy(pred,target_captions)

    return caption_loss

def pem_cls_loss_func(pred_score, gt_iou_map, mask):
    gt_iou_map = gt_iou_map * mask
    pmask = (gt_iou_map > 0.9).float()
    nmask = (gt_iou_map <= 0.9).float()
    nmask = nmask * mask

    num_positive = torch.sum(pmask)
    num_entries = num_positive + torch.sum(nmask)
    ratio = num_entries / num_positive
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    epsilon = 0.000001
    loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
    loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * nmask
    loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
    return loss