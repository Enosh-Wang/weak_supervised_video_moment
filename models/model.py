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
from DC.modules.modulated_deform_conv import ModulatedDeformConvPack

import inspect
from gpu_mem_track import MemTracker

class Model(nn.Module):

    def __init__(self, opt, vocab):
        super().__init__()
        self.opt = opt
        self.vocab = vocab
        self.GRU = TextEncoderGRU(opt)
        self.loss = Criterion(opt)

    def forward(self, videos, words, w_len, writer, iters, lam):
        
        # 文本特征提取
        sentences, words, w_mask= self.GRU(words,w_len) # -> [b,c]

        # 创建正负样本对，模态交互
        triplet_loss, postive= self.loss(videos,words,w_mask,writer,iters,lam)

        return postive, triplet_loss