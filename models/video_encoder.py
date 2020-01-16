import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from collections import OrderedDict

class VideoEncoder(nn.Module):

    def __init__(self, video_dim, joint_dim):
        super(VideoEncoder, self).__init__()
        self.joint_dim = joint_dim

        self.fc1 = nn.Linear(video_dim*3, joint_dim*2)
        self.fc2 = nn.Linear(joint_dim*2, joint_dim)

        self.init_weights()

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
		

    def forward(self, videos, lengths_img):
        """Extract video feature vectors."""

        video_feature=self.fc1(videos) # weight [4096, 1024] feature [128, 14, 1024]
        video_feature = self.fc2(video_feature)

        return video_feature

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        # 可训练参数的词典
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        # 加载参数
        super(VideoEncoder, self).load_state_dict(new_state) # ['fc1.weight', 'fc1.bias', 'fc3.weight', 'fc3.bias']
