import torch
import torch.nn as nn
from models.IMRAM import frame_by_word
import torch.nn.functional as F
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

class scdm1(nn.Module):
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

        b,c,d,t = video.size()
        video = video.permute(0,2,3,1).view(b,-1,c) # ->[b,d*t,c]
        sentence = frame_by_word(video,mask.view(1,-1),sentence,sentence_mask,self.opt)

        gama = torch.tanh(self.ga(sentence))
        deta = torch.tanh(self.de(sentence))
        # plot_map(gama.permute(0,2,1).view(b,c,d,t).contiguous().cpu().detach().numpy(),'gama')
        # plot_map(deta.permute(0,2,1).view(b,c,d,t).contiguous().cpu().detach().numpy(),'deta')

        mean = torch.mean(video,dim=(1,2),keepdim=True)
        var = torch.var(video,dim=(1,2),keepdim=True) + 1e-6
        # plot_map(mean.unsqueeze(1).cpu().detach().numpy(),'mean')
        # plot_map(var.unsqueeze(1).cpu().detach().numpy(),'var')

        video = (video-mean)/var
        # plot_map(video.permute(0,2,1).view(b,c,d,t).contiguous().cpu().detach().numpy(),'video_norm')
        video = gama*video+deta
        # plot_map(video.permute(0,2,1).view(b,c,d,t).contiguous().cpu().detach().numpy(),'video_modu')
        video = video.permute(0,2,1).view(b,c,d,t)
        # exit()
        return video

class scdm(nn.Module):
    def __init__(self,opt):
        super().__init__()

        self.opt = opt

        self.fc = nn.Linear(opt.joint_dim, opt.joint_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.bn1 = nn.LayerNorm([opt.joint_dim,10,20])

    def forward(self,video,v_mask,words,w_mask):

        # w_mask [b,l]
        b,c,d,t = video.size()
        v_global = video.mean(dim = (2,3)).unsqueeze(-1) # [b,c,d,t] -> [b,c,1]
        attn = torch.bmm(words,v_global) # [b,l,c] [b,c,1] -> [b,l,1]
        attn = attn.masked_fill(w_mask.unsqueeze(-1) == 0,float('-inf'))
        attn = F.softmax(attn, dim=1)
        w_global = torch.bmm(words.transpose(1,2),attn) # [b,c,l] [b,l,1] -> [b,c,1]

        # ch_attn = torch.tanh(self.fc(torch.cat([w_global,v_global],dim=1).squeeze())).view(b,c,1,1) # [b,c]
        ch_attn = torch.tanh(self.fc(w_global.squeeze())).view(b,c,1,1) # [b,c]
        video = self.bn1(video)
        video = video*ch_attn

        return video