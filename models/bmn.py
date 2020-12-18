# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
from tools.util import l2norm
import torch.nn.functional as F
from collections import OrderedDict
from models.IMRAM import SCAN,ContrastiveLoss,frame_by_word

import random

from models.mapping import mapping
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def plot_map(score_maps, name):
    # 可视化保存路径
    path = os.path.join(name)
    if not os.path.exists(path):
        os.makedirs(path)
    
    batch_size = score_maps.shape[0]
    channel_size = score_maps.shape[1]
    f = plt.figure(figsize=(6,4))
    for i in range(1):
        for j in range(channel_size):
            score_map = score_maps[i,j]
            plt.matshow(score_map, cmap = plt.cm.cool)
            plt.ylabel("duration")
            plt.xlabel("start time")
            plt.colorbar()
            plt.savefig(os.path.join(path,str(i)+'_'+str(j)+'.png'))
            plt.clf()
    plt.close(f)

def plot_maps(score_maps, name):
    # 可视化保存路径
    path = os.path.join(name)
    if not os.path.exists(path):
        os.makedirs(path)
    
    f = plt.figure(figsize=(6,4))
    plt.matshow(score_maps, cmap = plt.cm.cool)
    plt.ylabel("b")
    plt.xlabel("c")
    plt.colorbar()
    plt.savefig(os.path.join(path,'sentence.png'))
    plt.clf()
    plt.close(f)

def plot_mask(score_map, name):
    # 可视化保存路径

    f = plt.figure(figsize=(6,4))
    plt.matshow(score_map, cmap = plt.cm.cool)
    plt.ylabel("y")
    plt.xlabel("x")
    plt.colorbar()
    plt.savefig(name+'.png')
    plt.close(f)

def mask2weight(mask2d, mask_kernel, padding=0):
    # from the feat2d.py,we can know the mask2d is 4-d
    #plot_mask(mask2d.cpu().detach().numpy(),'mask')
    weight = torch.conv2d(mask2d[None,None,:,:].float(),
        mask_kernel, padding=padding)[0, 0]
    #plot_mask(weight.cpu().detach().numpy(),'weight')
    weight[weight > 0] = 1 / weight[weight > 0]
    #plot_mask(weight.cpu().detach().numpy(),'re_weight')
    return weight

class BMN(nn.Module):
    def __init__(self, opt, match_map ,v_mask):
        super(BMN, self).__init__()
        self.opt = opt
        self.tscale = opt.temporal_scale # 时序尺寸（论文中的T）
        self.prop_boundary_ratio = opt.prop_boundary_ratio # proposal拓展比率
        self.duration_start = int(self.tscale*opt.start_ratio)
        self.duration_end = int(self.tscale*(1-opt.end_ratio))
        self.num_sample = opt.num_sample # 采样点数目（论文中的N）
        self.num_sample_perbin = opt.num_sample_perbin # 子采样点的数目
        self.length, self.center = self.get_pmap(match_map)
        self._get_interp1d_mask()
        
        # Proposal Evaluation Module
        self.conv_1d = nn.Sequential(
            nn.Conv1d(opt.video_dim, opt.joint_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(opt.joint_dim),
            nn.ReLU(inplace=True)
        )
        self.conv_3d = nn.Sequential(
            nn.Conv3d(opt.joint_dim, opt.joint_dim, kernel_size=(self.num_sample, 1, 1),stride=(self.num_sample, 1, 1)),
            nn.BatchNorm3d(opt.joint_dim),
            nn.ReLU(inplace=True)
        )
        self.conv_2d = nn.Sequential(
            nn.Conv2d(opt.joint_dim+2, opt.joint_dim, kernel_size=1),
            nn.BatchNorm2d(opt.joint_dim),
            nn.ReLU(inplace=True)
        )
        
        k = opt.kernel_size
        l = opt.map_layers

        mask_kernel = torch.ones(1,1,k,k).cuda()
        first_padding = (k - 1) * l // 2

        self.weights = [
            mask2weight(v_mask, mask_kernel, padding=first_padding) 
        ]
        self.conv_map = nn.ModuleList(
            [nn.Conv2d(opt.joint_dim, opt.joint_dim, k, padding=first_padding)]
        )

        for _ in range(l - 1):
            self.weights.append(mask2weight(self.weights[-1] > 0, mask_kernel))
            self.conv_map.append(nn.Conv2d(opt.joint_dim, opt.joint_dim, k))

        for m in self.modules():
            if isinstance(m, (nn.Conv1d,nn.Conv2d,nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu') # 卷积层参数初始化
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # self.mdc = ModulatedDeformConvPack(self.hidden_dim_2d,self.opt.joint_dim,kernel_size=3,stride=1,padding=1)

    def forward(self, video, words, sentences, v_mask, w_mask):
        # 转换维度，维度在前，帧数在后 [b,c,t] sentence [b,c]
        video = video.transpose(1,2)

        v_map = self.conv_1d(video)
        v_map = self._boundary_matching_layer(v_map)
        v_map = self.conv_3d(v_map).squeeze(2) # -> [b,c,d,t]
        
        batch_size = v_map.size(0)
        center = self.center.repeat(batch_size,1,1,1)
        length = self.length.repeat(batch_size,1,1,1)
        v_map = torch.cat([v_map,center,length],dim=1)
        # plot_map(v_map.cpu().detach().numpy(),'pe')
        v_map = self.conv_2d(v_map)*v_mask.float()
        # plot_map(v_map.cpu().detach().numpy(),'pe_conv')
        for layer, weight in zip(self.conv_map, self.weights):
            v_map = layer(v_map).relu() * weight
            # plot_map((v_map*v_mask.float()).cpu().detach().numpy(),'scdm_weight_mask')
            # exit()
        return v_map

    def get_pmap(self, match_map):

        match_map = torch.Tensor(match_map).cuda().view(-1,self.tscale,2)
        match_map = match_map.permute(2,0,1)

        length = match_map[1,:,:] - match_map[0,:,:]
        center = match_map[0,:,:] + length/2
        
        return length.view(1,1,-1,self.tscale), center.view(1,1,-1,self.tscale)

    def _boundary_matching_layer(self, x):
        input_size = x.size()
        # out [b,c,n,d,t]
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0],input_size[1],self.num_sample,self.duration_end-self.duration_start,self.tscale)
        return out

    # 输入：sample_xmin, sample_xmax, self.tscale, self.num_sample,self.num_sample_perbin
    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        # Proposal的长度
        plen = float(seg_xmax - seg_xmin)
        # 采样点的长度
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        # 采样点的下标
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]

        p_mask = []
        # 迭代每个采样点，NxT矩阵的每一行
        for idx in range(num_sample):
            # 每个子采样点
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]

            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                # 向上取整
                sample_upper = math.ceil(sample)
                # 返回整数和小数部分
                sample_decimal, sample_down = math.modf(sample)
                # 参考论文采样公式
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            # 对子采样点取均值
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        # N x T 的矩阵
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for start_index in range(self.tscale):
            mask_mat_vector = []
            for duration_index in range(self.duration_start,self.duration_end):
                # 如果超过视频长度，则0填充
                if start_index + duration_index < self.tscale:
                    # 起始位置
                    p_xmin = start_index
                    # 结束位置
                    p_xmax = start_index + duration_index
                    # 持续时间长度，不就是duration_index+？？
                    center_len = float(p_xmax - p_xmin) + 1
                    # 扩展proposal
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    # 
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.tscale, self.num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                # p_mask[t,n]矩阵
                mask_mat_vector.append(p_mask)
            # mask_mat_vetor[t,n,d]
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        # mask_mat_vetor[t,n,d,t]
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        # 应当是转换成模型的一部分，以作缓存
        self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.tscale, -1), requires_grad=False)



