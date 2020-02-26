import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from collections import OrderedDict
from tools.util import PositionalEncoding,multihead_mask

class VideoEncoder(nn.Module):

    def __init__(self, opt):
        super(VideoEncoder, self).__init__()

        self.opt = opt
        self.fc1 = nn.Linear(opt.video_dim, opt.joint_dim)
        self.PE = PositionalEncoding(d_model = opt.joint_dim,dropout=opt.dropout,max_len=1000)
        self.norm = nn.BatchNorm1d(opt.joint_dim)
        self.attention = nn.ModuleList([nn.MultiheadAttention(embed_dim=opt.joint_dim,num_heads=opt.video_heads) for _ in range(opt.video_attn_layers)])

    def forward(self, videos, video_lengths):
        """Extract video feature vectors."""
        videos = videos.transpose(0,1)
        videos = self.fc1(videos)
        mask = multihead_mask(videos,video_lengths)
        videos = self.PE(videos)

        for layer in self.attention:
            res = videos
            videos, _ = layer(videos,videos,videos,mask)
            videos = F.dropout(videos,self.opt.dropout,self.training)
            videos = videos + res

        videos = self.norm(videos.permute(1,2,0))
        videos = videos.transpose(1,2)
        return videos,mask

