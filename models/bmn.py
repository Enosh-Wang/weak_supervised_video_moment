# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
from tools.util import l2norm
import torch.nn.functional as F
from collections import OrderedDict
from models.IMRAM import frame_by_word
from DC.modules.modulated_deform_conv import ModulatedDeformConvPack

class BMN(nn.Module):
    def __init__(self, opt):
        super(BMN, self).__init__()
        self.opt = opt
        self.tscale = opt.temporal_scale # 时序尺寸（论文中的T）
        self.prop_boundary_ratio = opt.prop_boundary_ratio # proposal拓展比率
        self.duration_start = int(self.tscale*opt.start_ratio)
        self.duration_end = int(self.tscale*(1-opt.end_ratio))
        self.num_sample = opt.num_sample # 采样点数目（论文中的N）
        self.num_sample_perbin = opt.num_sample_perbin # 子采样点的数目

        self._get_interp1d_mask()

        self.hidden_dim_1d = 512
        self.hidden_dim_2d = 512
        self.hidden_dim_3d = 512

        # Proposal Evaluation Module
        self.x_1d_p = nn.Sequential(OrderedDict([
            ('conv1',nn.Conv1d(400, self.hidden_dim_1d, kernel_size=3, padding=1)),
            ('norm1',nn.BatchNorm1d(self.hidden_dim_1d)),
            ('relu1',nn.ReLU(inplace=True))])
        )
        self.x_3d_p = nn.Sequential(OrderedDict([
            ('conv2',nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1),stride=(self.num_sample, 1, 1))),
            ('norm2',nn.BatchNorm3d(self.hidden_dim_3d)),
            ('relu2',nn.ReLU(inplace=True))])
        )
        self.x_2d_p1 = nn.Sequential(OrderedDict([
            ('conv3',nn.Conv2d(self.hidden_dim_3d+2, self.hidden_dim_2d, kernel_size=1)), 
            ('norm3',nn.BatchNorm2d(self.hidden_dim_2d)),
            ('relu3',nn.ReLU(inplace=True))])
        )
        self.x_2d_p2 = nn.Sequential(OrderedDict([
            ('conv4',nn.Conv2d(self.hidden_dim_2d, self.opt.joint_dim, kernel_size=3,padding=1)), 
            ('norm4',nn.BatchNorm2d(self.opt.joint_dim)),
            ('relu4',nn.ReLU(inplace=True))])
        )
        self.x_2d_p3 = nn.Sequential(OrderedDict([
            ('conv5',nn.Conv2d(self.opt.joint_dim*2, self.opt.joint_dim, kernel_size=1)), 
            ('norm5',nn.BatchNorm2d(self.opt.joint_dim)),
            ('relu5',nn.ReLU(inplace=True))])
        )

        # for m in self.modules():
        #     if isinstance(m,nn.Conv1d) or isinstance(m,nn.Conv2d) or isinstance(m,nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, nonlinearity='relu')#卷积层参数初始化
        #         nn.init.zeros_(m.bias)
        #     elif isinstance(m,nn.BatchNorm1d) or isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.BatchNorm3d):
        #         nn.init.ones_(m.weight)
        #         nn.init.zeros_(m.bias)
        for n,m in self.named_modules():
            if 'conv' in n:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')#卷积层参数初始化
                nn.init.zeros_(m.bias)
            if 'norm' in n:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.mdc = ModulatedDeformConvPack(self.opt.joint_dim,1,kernel_size=3,stride=1,padding=1)


    def forward(self, video, sentence, mask, match_map, sentence_mask):
        # 转换维度，维度在前，帧数在后 [b,c,t] sentence [b,c]
        video = video.transpose(1,2)
        # 一层时序卷积
        confidence_map = self.x_1d_p(video)
        # 置信度图
        confidence_map = self._boundary_matching_layer(confidence_map)
        confidence_map = self.x_3d_p(confidence_map).squeeze(2) # -> [b,c,d,t]
        
        # 拼接时间戳
        b,c,d,t = confidence_map.size()
        match_map = torch.Tensor(match_map).cuda().view(1,d,t,2)
        match_map = match_map.permute(0,3,1,2).repeat(b,1,1,1)
        length = match_map[:,1,:,:] - match_map[:,0,:,:]
        center = match_map[:,0,:,:] + length/2
        #confidence_map = torch.cat((confidence_map,match_map),dim=1) # 拼接开始和结束时刻
        confidence_map = torch.cat((confidence_map,center.unsqueeze(1),length.unsqueeze(1)),dim=1) # 拼接中点和长度
        confidence_map = self.x_2d_p1(confidence_map)

        output = self.x_2d_p2(confidence_map)
        
        # # 模态处理
        # confidence_map = confidence_map.permute(0,2,3,1).view(b,-1,c) # ->[b,d*t,c]
        # sentence = frame_by_word(confidence_map,mask.view(1,-1),sentence,sentence_mask,self.opt)
        # confidence_map = confidence_map.permute(0,2,1).view(b,c,d,t)
        # sentence = sentence.permute(0,2,1).view(b,c,d,t)
        # feature = torch.cat([confidence_map,sentence],dim=1)
        # feature = self.x_2d_p3(feature)

        attn = torch.sigmoid(self.mdc(confidence_map))
        #feature = self.x_2d_p3(feature)
        output = output*attn
        
        return output

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



