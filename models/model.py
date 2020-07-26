import torch
import torch.nn as nn
import torch.nn.functional as F
from models.text_encoder import TextEncoderMulti,TextEncoderGRU,TextNet
from models.video_caption import VideoCaption
from models.video_encoder import VideoEncoder
from models.loss import Criterion,Caption_Criterion,Criterion1,pem_cls_loss_func,Contrastive
from models.tanh_attention import TanhAttention
from models.mli import MLI
from models.bmn import BMN
from models.bilinear import Bilinear
import numpy as np
from tools.util import get_mask, get_mask_spare,get_iou_map,iou_with_anchors,get_match_map,Lognorm
from models.cross import xattn_score_t2i
from models.IMRAM import SCAN,ContrastiveLoss,frame_by_word
from models.bmn_action import BMN_action


class Model(nn.Module):

    def __init__(self, opt, vocab):
        super().__init__()
        self.opt = opt
        self.vocab = vocab
        self.GRU = TextEncoderGRU(opt)
        self.bmn = BMN(opt)
        #self.bmn_action = BMN_action(opt)
        self.mask = torch.IntTensor(get_mask(opt.temporal_scale,opt.start_ratio,opt.end_ratio)).cuda()#.view(1,-1)
        self.valid_num = torch.sum(self.mask)
        self.match_map = get_match_map(opt.temporal_scale,opt.start_ratio,opt.end_ratio)
        self.iou_map = get_iou_map(opt.temporal_scale,opt.start_ratio,opt.end_ratio,self.match_map,self.mask.view(1,-1).cpu().numpy())

        self.fc = nn.Linear(opt.video_dim, 400)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)   

    def forward(self, videos, sentences, sentence_lengths, word_id, gt_iou_map, writer, iters, lam):

        global_sentences, sentence_embedding, sentence_mask= self.GRU(sentences,sentence_lengths) # -> [b,c]

        videos = self.fc(videos)

        video_embedding_bmn = self.bmn(videos,sentence_embedding,self.mask,self.match_map,sentence_mask) # -> [b,c,d,t]
        
        triplet_loss, postive= Criterion(video_embedding_bmn,global_sentences,self.opt,writer,iters,self.training,lam,self.mask,self.match_map, self.valid_num, self.iou_map)
       
        # if self.training and iters % self.opt.log_step == 0:
        #     writer.add_scalar('triplet_loss',triplet_loss,iters)

        return postive, triplet_loss
