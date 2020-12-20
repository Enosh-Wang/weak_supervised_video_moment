import torch
import torch.nn as nn
from models.SCDM import scdm

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



class mapping(nn.Module):
    def __init__(self, opt, in_channels, out_channels , kernel_size, padding = 0):
        super(mapping, self).__init__()
        self.opt = opt
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding)
        # self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding)
        # self.norm = nn.BatchNorm2d(out_channels)
        self.modulation = scdm(opt)

    def forward(self, v_map, v_mask, words, w_mask):
        v_map = self.modulation(v_map,v_mask,words,w_mask)
        # plot_map(v_map.cpu().detach().numpy(),'scdm')
        v_map = self.conv(v_map).relu()
        # v_map = self.conv1(v_map).relu()
        # plot_map(v_map.cpu().detach().numpy(),'scdm_conv')
        
        return v_map