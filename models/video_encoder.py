import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from collections import OrderedDict
from tools.util import PositionalEncoding,multihead_mask,l2norm

class VideoEncoder(nn.Module):

    def __init__(self, opt):
        super(VideoEncoder, self).__init__()

        self.opt = opt
        self.fc1 = nn.Linear(opt.video_dim, opt.joint_dim)
        self.PE = PositionalEncoding(d_model = opt.joint_dim,dropout=opt.dropout,max_len=1000)
        self.norm = nn.BatchNorm1d(opt.joint_dim) #[b,c,l]
        self.attention = nn.ModuleList([nn.MultiheadAttention(embed_dim=opt.joint_dim,num_heads=opt.video_heads) for _ in range(opt.video_attn_layers)])

    def forward(self, videos):
        """Extract video feature vectors."""

        videos = self.fc1(videos).transpose(1,2) # [b,l,c]->[b,c,l]
        videos = self.norm(videos).permute(2,0,1) # [b,c,l]->[l,b,c]
        videos = self.PE(videos)

        for layer in self.attention:
            res = videos
            videos, _ = layer(videos,videos,videos)
            videos = F.dropout(videos,self.opt.dropout,self.training)
            videos = videos + res

        videos = videos.transpose(0,1) # [l,b,c]->[b,l,c]
        videos = l2norm(videos, dim=2)

        return videos

class VideoLinearEncoder(nn.Module):

    def __init__(self, opt):
        super(VideoLinearEncoder, self).__init__()

        self.opt = opt
        self.fc1 = nn.Linear(opt.video_dim, opt.joint_dim)
        self.PE = PositionalEncoding(d_model = opt.joint_dim,dropout=opt.dropout,max_len=1000)
        self.norm = nn.BatchNorm1d(opt.joint_dim) #[b,c,l]
        self.attention = nn.ModuleList([nn.MultiheadAttention(embed_dim=opt.joint_dim,num_heads=opt.video_heads) for _ in range(opt.video_attn_layers)])

    def forward(self, videos):
        """Extract video feature vectors."""

        videos = self.fc1(videos).transpose(1,2) # [b,l,c]->[b,c,l]
        videos = self.norm(videos).permute(2,0,1) # [b,c,l]->[l,b,c]
        videos = self.PE(videos)

        for layer in self.attention:
            res = videos
            videos, _ = layer(videos,videos,videos)
            videos = F.dropout(videos,self.opt.dropout,self.training)
            videos = videos + res

        videos = videos.transpose(0,1) # [l,b,c]->[b,l,c]
        videos = l2norm(videos, dim=2)

        return videos
