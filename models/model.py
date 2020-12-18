import torch
import torch.nn as nn
import torch.nn.functional as F
from models.text_encoder import TextEncoderMulti,TextEncoderGRU,TextNet
from models.video_caption import VideoCaption
from models.video_encoder import VideoEncoder
from models.loss import Criterion
from models.tanh_attention import TanhAttention
from models.mli import MLI
from models.bmn import BMN
from models.bilinear import Bilinear
import numpy as np
from tools.util import get_mask, get_mask_spare,get_iou_map,iou_with_anchors,get_match_map,Lognorm
from models.cross import xattn_score_t2i
from models.IMRAM import SCAN,ContrastiveLoss,frame_by_word
from models.bmn_action import BMN_action
<<<<<<< HEAD
#from DC.modules.modulated_deform_conv import ModulatedDeformConvPack
=======
# from DC.modules.modulated_deform_conv import ModulatedDeformConvPack
>>>>>>> b39d49fd1a7336ed7eb4478e38a14def76ad2247

class Model(nn.Module):

    def __init__(self, opt, vocab):
        super().__init__()
        self.opt = opt
        self.vocab = vocab
        self.GRU = TextEncoderGRU(opt)
        
<<<<<<< HEAD
        self.v_mask = torch.IntTensor(get_mask(opt.temporal_scale,opt.start_ratio,opt.end_ratio)).cuda()#.view(1,-1)
        self.valid_num = torch.sum(self.v_mask)
        self.match_map = get_match_map(opt.temporal_scale,opt.start_ratio,opt.end_ratio)
        self.iou_map = get_iou_map(opt.temporal_scale,opt.start_ratio,opt.end_ratio,self.match_map,self.v_mask.view(1,-1).cpu().numpy())
        self.bmn = BMN(opt, self.match_map,self.v_mask.float())
        self.loss = Criterion(opt,self.v_mask)

    def forward(self, videos, words, w_len, writer, iters, lam):
=======
        #self.bmn_action = BMN_action(opt)
        self.mask = torch.IntTensor(get_mask(opt.temporal_scale,opt.start_ratio,opt.end_ratio)).cuda()#.view(1,-1)
        self.valid_num = torch.sum(self.mask)
        self.match_map = get_match_map(opt.temporal_scale,opt.start_ratio,opt.end_ratio)
        self.iou_map = get_iou_map(opt.temporal_scale,opt.start_ratio,opt.end_ratio,self.match_map,self.mask.view(1,-1).cpu().numpy())
        self.bmn = BMN(opt, self.match_map,self.mask.float())
        # self.scan = SCAN(opt)
        # self.loss = ContrastiveLoss(opt)
        #self.loss = Contrastive(opt)
        # self.fc = nn.Linear(opt.video_dim, opt.joint_dim)
        # nn.init.xavier_uniform_(self.fc.weight)
        # nn.init.zeros_(self.fc.bias)
        

    def forward(self, videos, sentences, sentence_lengths, word_id, writer, iters, lam):
>>>>>>> b39d49fd1a7336ed7eb4478e38a14def76ad2247

        # 文本特征提取
        sentences, words, w_mask= self.GRU(words,w_len) # -> [b,c]

<<<<<<< HEAD
        # 视频特征提取
        v_map = self.bmn(videos, self.v_mask) # -> [b,c,d,t]
        
        # 创建正负样本对，模态交互
        triplet_loss, postive= self.loss(v_map,words,w_mask,writer,iters,lam,self.v_mask,self.valid_num, self.iou_map)

    
        # if self.training and iters % self.opt.log_step == 0:
        #     writer.add_scalar('self_loss',loss,iters)

=======
        videos = torch.ones(videos.size(0),videos.size(1),self.opt.joint_dim).cuda()#= self.fc(videos)

        video_embedding_bmn = self.bmn(videos,sentence_embedding,global_sentences,self.mask,sentence_mask) # -> [b,c,d,t]
    
        
        triplet_loss, postive= Criterion(video_embedding_bmn,global_sentences,self.opt,writer,iters,self.training,lam,self.mask,self.match_map, self.valid_num, self.iou_map)


>>>>>>> b39d49fd1a7336ed7eb4478e38a14def76ad2247
        return postive, triplet_loss