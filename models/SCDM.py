import torch
import torch.nn as nn
from models.IMRAM import frame_by_word

import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import inspect
from gpu_mem_track import  MemTracker

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

class scdm(nn.Module):
    def __init__(self,opt):
        super().__init__()

        self.opt = opt

        self.ga = nn.Linear(opt.joint_dim, opt.joint_dim)
        nn.init.xavier_uniform_(self.ga.weight)
        nn.init.zeros_(self.ga.bias)

        self.de = nn.Linear(opt.joint_dim, opt.joint_dim)
        nn.init.xavier_uniform_(self.de.weight)
        nn.init.zeros_(self.de.bias)

    def forward(self,video,mask,sentence,sentence_mask):
        frame = inspect.currentframe()          # define a frame to track
        gpu_tracker = MemTracker(frame)         # define a GPU tracker
        gpu_tracker.track()
        b,c,d,t = video.size()
        video = video.permute(0,2,3,1).view(b,-1,c) # ->[b,d*t,c]
        sentence = frame_by_word(video,mask.view(1,-1),sentence,sentence_mask,self.opt)
        gpu_tracker.track()
        gama = torch.tanh(self.ga(sentence))
        deta = torch.tanh(self.de(sentence))
        # plot_map(gama.permute(0,2,1).view(b,c,d,t).contiguous().cpu().detach().numpy(),'gama')
        # plot_map(deta.permute(0,2,1).view(b,c,d,t).contiguous().cpu().detach().numpy(),'deta')
        gpu_tracker.track()
        mean = torch.mean(video,dim=(1,2),keepdim=True)
        var = torch.var(video,dim=(1,2),keepdim=True) + 1e-6
        # plot_map(mean.unsqueeze(1).cpu().detach().numpy(),'mean')
        # plot_map(var.unsqueeze(1).cpu().detach().numpy(),'var')
        gpu_tracker.track()
        video = (video-mean)/var
        # plot_map(video.permute(0,2,1).view(b,c,d,t).contiguous().cpu().detach().numpy(),'video_norm')
        video = gama*video+deta
        # plot_map(video.permute(0,2,1).view(b,c,d,t).contiguous().cpu().detach().numpy(),'video_modu')
        video = video.permute(0,2,1).view(b,c,d,t)
        gpu_tracker.track()
        # exit()
        return video