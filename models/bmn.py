# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
from tools.util import l2norm

class BMN(nn.Module):
    def __init__(self, opt):
        super(BMN, self).__init__()
        self.opt = opt
        self.tscale = opt.temporal_scale # 时序尺寸（论文中的T）
        self.prop_boundary_ratio = opt.prop_boundary_ratio # proposal拓展比率
        self.num_sample = opt.num_sample # 采样点数目（论文中的N）
        self.num_sample_perbin = opt.num_sample_perbin # 子采样点的数目

        self._get_interp1d_mask()

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 256
        self.hidden_dim_3d = 512

        # Proposal Evaluation Module
        self.x_1d_p = nn.Sequential(
            nn.Conv1d(self.opt.joint_dim, self.hidden_dim_1d, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hidden_dim_1d),
            nn.ReLU(inplace=True)
        )
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1),stride=(self.num_sample, 1, 1)),
            nn.BatchNorm3d(self.hidden_dim_3d),
            nn.ReLU(inplace=True)
        )

        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1), 
            nn.BatchNorm2d(self.hidden_dim_2d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.opt.joint_dim, kernel_size=1), 
            nn.BatchNorm2d(self.opt.joint_dim),
            nn.Tanh()
        )

        

    def forward(self, video): #, sentence):
        # 转换维度，维度在前，帧数在后 [b,c,t]
        video = video.transpose(1,2)
        # 一层时序卷积
        confidence_map = self.x_1d_p(video)
        # 置信度图
        confidence_map = self._boundary_matching_layer(confidence_map)
        confidence_map = self.x_3d_p(confidence_map).squeeze(2) # -> [b,c,d,t]
        confidence_map = self.x_2d_p(confidence_map) # -> [b,c,d,t]
        
        confidence_map = l2norm(confidence_map,dim=1)
        return confidence_map

    def _boundary_matching_layer(self, x):
        input_size = x.size()
        # out [b,c,n,d,t]
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0],input_size[1],self.num_sample,self.tscale,self.tscale)
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
            for duration_index in range(self.tscale):
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



