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
from tools.util import cosine_similarity
import random
import numpy as np

class Model(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.video_encoder = VideoEncoder(opt)
        self.text_encoder = TextEncoderMulti(opt)
        self.GRU = TextEncoderGRU(opt)
        self.mli = MLI(opt)
        self.fc = nn.Linear(opt.joint_dim*2,1)
        self.bmn = BMN(opt)
        self.real_batch = opt.batch_size
        self.real_negative = opt.negative_num
    
    def forward(self, videos, sentences, sentence_lengths, video_lengths, video_name, is_training):

        #sentence_embedding, sentence_mask = self.text_encoder(sentences, sentence_lengths)
        self.real_batch = videos.size(0)
        sentence_embedding = self.GRU(sentences,sentence_lengths)
        video_embedding, video_mask= self.video_encoder(videos, video_lengths)
        #sentence_mask = torch.ones(sentence_embedding.size(0),1).bool().cuda()
        #video_embedding,sentence_embedding = self.mli(video_embedding, video_mask,sentence_embedding.unsqueeze(1), sentence_mask)
        #sentence_embedding = torch.mean(sentence_embedding,dim=1)  
        #sentence_embedding = sentence_embedding.squeeze(1)
        # video_embedding：[128,20,1024] sentence_embedding：[128,1024]
        # similarity = cosine_similarity(video_embedding, sentence_embedding)
        # confidence_map[128,20,20]
        video_embedding = self.bmn(video_embedding)
        negative_sentence,negative_video = self.sample(video_embedding,sentence_embedding,video_name)
        all_video = torch.cat((video_embedding,negative_video),dim=0)
        all_sentence = torch.cat((sentence_embedding,negative_sentence),dim=0)

        mask = self.get_mask(self.opt.temporal_scale).unsqueeze(0)
        score = self.cosine_similarity(all_video, all_sentence)
        score = score.masked_fill(mask == 0, float('-inf'))
        all_size = score.size(0)
        score = score.view(all_size,-1)
        #score = F.softmax(score,dim=-1)
        # score [b,d,t]
        loss = Criterion(score,self.real_batch,self.real_negative,self.opt)
        score = score.view(all_size,self.opt.temporal_scale,self.opt.temporal_scale)
        return score[:self.real_batch],loss

    def cosine_similarity(self, x1, x2):
        """Returns cosine similarity based attention between x1 and x2, computed along dim."""

        b,c,d,t = x1.size()
        x2 = x2.view(b,c,1,1)
        x2 = x2.repeat(1,1,d,t) 
        score = torch.cosine_similarity(x1, x2, dim=1)
        #score = torch.sum(x1*x2,dim=1)

        return score

    def get_mask(self, tscale):

        bm_mask = []
        # 遍历每行，逐行计算掩码，逐行在末尾增加0
        for idx in range(tscale):
            mask_vector = [1 for i in range(tscale - idx)
                        ] + [0 for i in range(idx)]
            bm_mask.append(mask_vector)
        bm_mask = np.array(bm_mask, dtype=np.float32)
        return torch.Tensor(bm_mask).cuda()

    def get_mask_spare(self, tscale):
         
        bm_mask = []
        # 遍历每行，逐行计算掩码，逐行在末尾增加0
        for idx in range(tscale):
            mask_vector = []
            for d in range(tscale - idx):
                s = np.ceil(d/16)
                if np.mod(idx,s) ==0 and np.mod(idx+np.floor(d/16),s) == 0:
                    mask_vector.append(1)
                else:
                    mask_vector.append(0)
            mask_vector += [0 for i in range(idx)]
            bm_mask.append(mask_vector)
        bm_mask = np.array(bm_mask, dtype=np.float32)
        return torch.Tensor(bm_mask).cuda()

    def sample(self,video_embedding,sentence_embedding,video_name):
        self.real_negative = min(self.opt.negative_num,self.real_batch)-1
        b,c,d,t = video_embedding.size()
        new_size = b*self.real_negative
        new_video = torch.zeros(new_size,c,d,t).cuda()
        new_sentence = torch.zeros(new_size,c).cuda()
        
        for i in range(b):
            x = list(range(b))
            x.remove(i)
            real_size = len(x)
            new_sentence[i*self.real_negative:(i+1)*self.real_negative] = sentence_embedding[i]
            # 至多重复采用5次，防止采样到与gt相同的视频
            for j in range(5):
                y = random.sample(x,self.real_negative)
                if video_name[i] not in [video_name[k] for k in y]:
                    break
            new_video[i*self.real_negative:(i+1)*self.real_negative] = video_embedding[y]
        return new_sentence,new_video
            



