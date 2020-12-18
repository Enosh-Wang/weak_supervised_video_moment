# -*- coding: utf-8 -*-

import random
import numpy as np
from scipy.interpolate import interp1d
import h5py
import os
import pickle


def poolData(feats,num_prop=100,num_bin=1,num_sample_bin=3,pool_type="mean"):
    #feats = feats[0::2]
    feat_length,feat_dim = feats.shape
    # 如果feature长度为1，直接堆叠即可
    if feat_length == 1:
        video_feature=np.stack([feats]*num_prop)
        video_feature=np.reshape(video_feature,[num_prop,feat_dim])
        return video_feature,feat_length

    # 线性插值函数(这里简化了一下，把特征的一帧看做一秒)
    x = [0.5+i for i in range(feat_length)]
    f=interp1d(x,feats,axis=0)
        
    video_feature=[]
    zero_sample=np.zeros(num_bin*feat_dim)
    # 100个anchor的下标和上标
    tmp_anchor_xmin=[1.0/num_prop*i for i in range(num_prop)] #[0-0.99]
    tmp_anchor_xmax=[1.0/num_prop*i for i in range(1,num_prop+1)] # [0.01-1.00]        
    # 3
    num_sample=num_bin*num_sample_bin
    # 遍历100个anchor
    for idx in range(num_prop):
        xmin=max(x[0]+0.0001,tmp_anchor_xmin[idx]*feat_length)
        xmax=min(x[-1]-0.0001,tmp_anchor_xmax[idx]*feat_length)
        if xmax<x[0]:
            video_feature.append(zero_sample)
            continue
        if xmin>x[-1]:
            video_feature.append(zero_sample)
            continue
        
        # 每个anchor采样3个点，线性插值
        plen=(xmax-xmin)/(num_sample-1)
        x_new=[xmin+plen*ii for ii in range(num_sample)]
        y_new=f(x_new)

        y_new_pool=[]
        for b in range(num_bin):
            tmp_y_new=y_new[num_sample_bin*b:num_sample_bin*(b+1)]
            if pool_type=="mean":
                tmp_y_new=np.mean(y_new,axis=0)
            elif pool_type=="max":
                tmp_y_new=np.max(y_new,axis=0)
            y_new_pool.append(tmp_y_new)
        y_new_pool=np.stack(y_new_pool)
        y_new_pool=np.reshape(y_new_pool,[-1])
        video_feature.append(y_new_pool)
    video_feature=np.stack(video_feature)
    return video_feature,feat_length
if __name__ == "__main__":
    
    datapath = '/home/share/wangyunxiao/TACoS'
    feature_path = 'tall_c3d_features.hdf5'
    video_feature = h5py.File(os.path.join(datapath,feature_path), 'r')

    video_list = []
    feats_list = []
    length = 0
    num = 0
    for key,value in video_feature.items():
        video = key
        feats = value
        # 把视频采样成100个点，每个点处的值又3个采样点取均值算得
        videoFeature_mean,feat_length=poolData(feats,num_prop=128,num_bin=1,num_sample_bin=3,pool_type="mean")
        length += feat_length
        num += 1
        video_list.append(video)
        feats_list.append(videoFeature_mean)
    print('average_length:',length/num)
    data = dict(zip(video_list,feats_list))
    # 保存新的特征
    with open(os.path.join(datapath,'tacos_128.pkl'),'wb') as f:
        pickle.dump(data,f)
