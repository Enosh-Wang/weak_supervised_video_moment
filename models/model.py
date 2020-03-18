import torch
import torch.nn as nn
import torch.nn.functional as F
from models.text_encoder import TextEncoderMulti,TextEncoderGRU
from models.video_caption import VideoCaption
from models.video_encoder import VideoEncoder
from models.loss import Criterion
from models.tanh_attention import TanhAttention
from models.mli import MLI
from models.bmn import BMN
from models.bilinear import Bilinear
import numpy as np
from tools.util import get_mask, get_mask_spare
from tools.plot_gt_map import get_match_map

class Model(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.video_encoder = VideoEncoder(opt)
        #self.text_encoder = TextEncoderMulti(opt)
        self.GRU = TextEncoderGRU(opt)
        #self.mli = MLI(opt)
        self.bmn = BMN(opt)
        self.mask = get_mask(opt.temporal_scale)
        self.valid_num = torch.sum(self.mask)
        self.match_map = get_match_map(opt.temporal_scale)
        #self.bilinear = Bilinear(opt.joint_dim,opt.joint_dim,opt.joint_dim)

    def forward(self, videos, sentences, sentence_lengths, video_name, writer, iters, max_iter):

        sentence_embedding = self.GRU(sentences,sentence_lengths) # -> [b,l,c]
        video_embedding = self.video_encoder(videos) # -> [b,l,c]

        video_embedding_bmn = self.bmn(video_embedding) # -> [b,c,d,t]

        loss, postive = Criterion(video_embedding_bmn,sentence_embedding,self.opt,writer,iters,self.training,max_iter,self.mask,self.match_map,video_name,self.valid_num)
        
        if self.training and iters % self.opt.log_step == 0:
            writer.add_histogram('sentence_embedding',sentence_embedding,iters)
            writer.add_histogram('video_embedding',video_embedding,iters)
            writer.add_histogram('video_embedding_bmn',video_embedding_bmn,iters)
            # writer.add_histogram('bmn/x_3d_p/bias',self.bmn.x_3d_p._modules['0'].bias,iters)
            # writer.add_histogram('bmn/x_3d_p/weight',self.bmn.x_3d_p._modules['0'].weight,iters)
            # writer.add_histogram('bmn/x_3d_p/norm_bias',self.bmn.x_3d_p._modules['1'].bias,iters)
            # writer.add_histogram('bmn/x_3d_p/norm_weight',self.bmn.x_3d_p._modules['1'].weight,iters)
            # writer.add_histogram('bmn/x_1d_p/bias',self.bmn.x_1d_p._modules['0'].bias,iters)
            # writer.add_histogram('bmn/x_1d_p/weight',self.bmn.x_1d_p._modules['0'].weight,iters)
            # writer.add_histogram('bmn/x_1d_p/norm_bias',self.bmn.x_1d_p._modules['1'].bias,iters)
            # writer.add_histogram('bmn/x_1d_p/norm_weight',self.bmn.x_1d_p._modules['1'].weight,iters)
            # writer.add_histogram('GRU/weight_ih_l0',self.GRU.rnn.weight_ih_l0,iters)
            # writer.add_histogram('GRU/weight_hh_l0',self.GRU.rnn.weight_hh_l0,iters)
            # writer.add_histogram('GRU/bias_ih_l0',self.GRU.rnn.bias_ih_l0,iters)
            # writer.add_histogram('GRU/bias_hh_l0',self.GRU.rnn.bias_hh_l0,iters)
            # writer.add_histogram('video_encoder/fc1/bias',self.video_encoder.fc1.bias,iters)
            # writer.add_histogram('video_encoder/fc1/weight',self.video_encoder.fc1.weight,iters)
            # if self.bmn.x_3d_p._modules['0'].weight.grad is not None:
            #     writer.add_histogram('bmn/x_3d_p/weight_grad',self.bmn.x_3d_p._modules['0'].weight.grad,iters)
            #     writer.add_histogram('bmn/x_3d_p/norm_weight_grad',self.bmn.x_3d_p._modules['1'].weight.grad,iters)
            #     writer.add_histogram('bmn/x_1d_p/weight_grad',self.bmn.x_1d_p._modules['0'].weight.grad,iters)
            #     writer.add_histogram('bmn/x_1d_p/norm_weight_grad',self.bmn.x_1d_p._modules['1'].weight.grad,iters)
            #     writer.add_histogram('GRU/weight_ih_l0_grad',self.GRU.rnn.weight_ih_l0.grad,iters)
            #     writer.add_histogram('GRU/weight_hh_l0_grad',self.GRU.rnn.weight_hh_l0.grad,iters)
            #     writer.add_histogram('video_encoder/fc1/weight_grad',self.video_encoder.fc1.weight.grad,iters)
        
        return postive, loss

            



