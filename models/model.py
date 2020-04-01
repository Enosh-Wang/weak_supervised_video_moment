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
from models.cross import xattn_score_t2i

class Model(nn.Module):

    def __init__(self, opt, vocab):
        super().__init__()
        self.opt = opt
        self.vocab = vocab
        self.video_encoder = VideoEncoder(opt)
        #self.text_encoder = TextEncoderMulti(opt)
        self.GRU = TextEncoderGRU(opt)
        #self.mli = MLI(opt)
        self.bmn = BMN(opt)
        self.mask = get_mask(opt.temporal_scale)
        self.valid_num = torch.sum(self.mask)
        self.match_map = get_match_map(opt.temporal_scale)
        #self.bilinear = Bilinear(opt.joint_dim,opt.joint_dim,opt.joint_dim)
        self.caption = VideoCaption(opt,vocab)

    def forward(self, videos, sentences, sentence_lengths, video_name, word_id, writer, iters, max_iter):

        global_sentences, sentence_embedding = self.GRU(sentences,sentence_lengths) # -> [b,c]
        #video_embedding = self.video_encoder(videos) # -> [b,l,c]

        video_embedding_bmn,attn = self.bmn(videos,global_sentences,self.mask) # -> [b,c,d,t]
        #video_embedding_bmn,attn = xattn_score_t2i(video_embedding_bmn,sentence_embedding,sentence_lengths,self.opt)
        pred = self.caption(video_embedding_bmn, sentence_embedding, sentence_lengths)

        loss = Criterion(pred,word_id,sentence_lengths)
        #loss, postive = Criterion(video_embedding_bmn,global_sentences,attn,self.opt,writer,iters,self.training,max_iter,self.mask,self.match_map,video_name,self.valid_num)
        
        if self.training and iters % self.opt.log_step == 0:
            writer.add_histogram('sentence_embedding',sentence_embedding,iters)
            #writer.add_histogram('video_embedding',video_embedding,iters)
            writer.add_histogram('video_embedding_bmn',video_embedding_bmn,iters)
        
        return attn, loss

            



