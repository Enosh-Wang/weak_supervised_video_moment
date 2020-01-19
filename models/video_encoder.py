import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from collections import OrderedDict
from utils.util import PositionalEncoding,multihead_mask

class VideoEncoder(nn.Module):

    def __init__(self, opt):
        super(VideoEncoder, self).__init__()

        self.opt = opt
        self.fc1 = nn.Linear(opt.video_dim, opt.joint_dim)
        self.PE = PositionalEncoding(d_model = opt.joint_dim,dropout=opt.dropout,max_len=1000)
        
        self.attention = nn.ModuleList([nn.MultiheadAttention(embed_dim=opt.joint_dim,num_heads=opt.video_heads) for _ in range(opt.video_attn_layers)])
        
        self.fc2 = nn.Linear(opt.joint_dim, opt.joint_dim)

        #self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
           同上
        """
        r1 = np.sqrt(6.) / np.sqrt(self.fc1.in_features +
                                  self.fc1.out_features)
        self.fc1.weight.data.uniform_(-r1, r1)
        self.fc1.bias.data.fill_(0)
        r2 = np.sqrt(6.) / np.sqrt(self.fc2.in_features +
                                  self.fc2.out_features)
        self.fc2.weight.data.uniform_(-r2, r2)
        self.fc2.bias.data.fill_(0)
		

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

        videos = videos.transpose(0,1)
        return videos,mask

