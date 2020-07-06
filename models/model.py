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
from tools.util import get_mask, get_mask_spare,get_iou_map,iou_with_anchors,get_match_map
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
        # self.scan = SCAN(opt)
        # self.loss = ContrastiveLoss(opt)
        self.loss = Contrastive(opt)
        self.fc = nn.Linear(opt.video_dim, 400)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        self.bert = TextNet(opt)

    def forward(self, videos, sentence_lengths, word_id, gt_iou_map, segments, input_masks, writer, iters, lam):

        sentence_embedding = self.bert(word_id, segments.cuda(), input_masks.cuda())
        global_sentences = torch.mean(sentence_embedding,dim=1)
        #global_sentences, sentence_embedding, sentence_mask= self.GRU(sentences,sentence_lengths) # -> [b,c]

        videos = self.fc(videos)
        # confidence_map, start, end = self.bmn_action(videos)
        # batch = videos.size(0)
        # confidence_map = (confidence_map[:,1,:,:]*self.mask.float()).view(batch,-1)

        video_embedding_bmn,confidence = self.bmn(videos,sentence_embedding,self.mask,self.match_map,input_masks.cuda()) # -> [b,c,d,t]
        
        #loss = pem_cls_loss_func(pre_video_embedding_bmn, gt_iou_map.cuda(), self.mask.float())
        #loss = F.mse_loss(confidence_map,(confidence.squeeze()*self.mask.float()).view(batch,-1))
        #loss = pem_cls_loss_func(video_embedding_bmn, gt_iou_map, self.mask.float())
        
        #score,attn = self.scan.forward_score(video_embedding_bmn,sentence_embedding,sentence_lengths)
        #loss = self.loss(video_embedding_bmn)

        #pred = self.caption(video_embedding_bmn, sentence_embedding, sentence_lengths)
        #caption_loss = Caption_Criterion(pred,word_id,sentence_lengths)
        
        triplet_loss, postive= self.loss(video_embedding_bmn,global_sentences,self.opt,writer,iters,self.training,lam,self.mask,self.match_map, self.valid_num, self.iou_map)

        
        # if self.training and iters % self.opt.log_step == 0:
        #     writer.add_scalar('caption_loss',caption_loss,iters)
        #     writer.add_scalar('triplet_loss',triplet_loss,iters)

        return postive, triplet_loss#+triplet_loss2#/10+caption_loss

            


# tscale = self.opt.temporal_scale
        # b_map = []
        # for i in range(video_embedding_bmn.size(0)):
        #     index = orders[i][0]
        #     start_index = index%tscale
        #     end_index = start_index + index//tscale + 1
        #     xmin = start_index/tscale
        #     xmax = end_index/tscale
            
        #     iou_map = iou_with_anchors(self.match_map[:, 0],self.match_map[:, 1],xmin,xmax)
        #     iou_map = np.reshape(iou_map, [tscale, tscale])
        #     iou_map = torch.Tensor(iou_map).cuda()
        #     b_map.append((iou_map >= 0.01).int()*self.mask) #[d,t]
        
        # b_map = torch.stack(b_map).unsqueeze(1) # [b,1,d,t]
        # print(torch.sum(b_map[0]))
        # video_embedding_bmn = video_embedding_bmn*b_map.float()
        # video_embedding_bmn = self.conv(video_embedding_bmn)

        # triplet_loss2, postive= Criterion1(video_embedding_bmn,global_sentences,self.opt,writer,iters,self.training,max_iter,b_map,self.match_map, self.valid_num)

