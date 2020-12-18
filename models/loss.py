import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tools.util import LogSumExp, get_mask
from models.CMIL import get_lambda, get_video_score_nms, get_video_score_nms_all
import time
import random
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

def plot_map(score_maps, map_name):
    score_maps = score_maps.cpu().numpy()

    batch_size = score_maps.shape[0]
    f = plt.figure(figsize=(6,4))
    for i in range(batch_size):
        score_map = score_maps[i]
        plt.matshow(score_map, cmap = plt.cm.cool)
        plt.ylabel("duration")
        plt.xlabel("start time")
        plt.colorbar()
        plt.savefig(os.path.join(str(i)+map_name+'.png'))
        plt.clf()    
    plt.close(f)

def norm(score):
    b,d,t = score.size()
    score = score.view(b,-1)
    score_min = torch.min(score, dim = -1, keepdim = True)[0]
    score_max = torch.max(score, dim = -1, keepdim = True)[0]
    score_norm = (score - score_min)/(score_max-score_min)
    score = score - score_min
    score = score*score_norm
    score = score.view(b,d,t)
    return score

class Contrastive(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt):
        super(Contrastive, self).__init__()
        self.opt = opt
        self.margin = opt.global_margin
        self.max_violation = opt.max_violation

        self.fc = nn.Linear(opt.joint_dim, opt.joint_dim//4)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        self.fc1 = nn.Linear(opt.joint_dim, opt.joint_dim//4)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(opt.joint_dim, opt.joint_dim//4)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        self.fc3 = nn.Linear(opt.joint_dim, opt.joint_dim//4)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

        self.fc4 = nn.Linear(opt.joint_dim, opt.joint_dim//4)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)

        self.fc5 = nn.Linear(opt.joint_dim, opt.joint_dim//4)
        nn.init.xavier_uniform_(self.fc5.weight)
        nn.init.zeros_(self.fc5.bias)

        self.fc6 = nn.Linear(opt.joint_dim, opt.joint_dim//4)
        nn.init.xavier_uniform_(self.fc6.weight)
        nn.init.zeros_(self.fc6.bias)

        self.fc7 = nn.Linear(opt.joint_dim, opt.joint_dim//4)
        nn.init.xavier_uniform_(self.fc7.weight)
        nn.init.zeros_(self.fc7.bias)

    def forward(self, videos, sentences, opt, writer, iters, is_training, lam, mask, match_map, valid_num, iou_maps):
        b,c,d,t = videos.size()
        postive_map = []
        score_map = []

        diversity_loss = []
        fb_loss = []
        vas_loss = []
        crov_loss = []

        #videos = videos.permute(0,2,3,1) # [b,c,d,t] -> [b,d,t,c]
        video_mean = torch.mean(videos,dim=1,keepdim=True)
        videos = videos - video_mean
        sentence_mean = torch.mean(sentences,dim=1,keepdim=True)
        sentences = sentences - sentence_mean

        videos = videos.permute(0,2,3,1) # [b,c,d,t] -> [b,d,t,c]
        videos1 = self.fc(videos).permute(0,3,1,2)
        videos2 = self.fc1(videos).permute(0,3,1,2)
        videos3 = self.fc2(videos).permute(0,3,1,2)
        videos4 = self.fc3(videos).permute(0,3,1,2)

        sentences1 = self.fc4(sentences)
        sentences2 = self.fc5(sentences)
        sentences3 = self.fc6(sentences)
        sentences4 = self.fc7(sentences)

        videos = torch.cat([videos1,videos2,videos3,videos4],dim=1)
        sentences = torch.cat([sentences1,sentences2,sentences3,sentences4],dim=1)
        # for i in range(b):
        #     # 聚合视频前景特征
        #     video = videos[i] # [b,c,d,t] -> [c,d,t]
        #     sentence = sentences[i].view(c,1,1).repeat(1,d,t)
        #     score = torch.cosine_similarity(video,sentence,dim=0).masked_fill(mask==0,float('-inf')).view(-1)
        #     score = F.softmax(score) # [d*t]
        #     video_postive = torch.sum(score*video.view(c,d*t),dim=1) # [c,d*t] -> [c]

        #     gt_sentence = sentences[i]
        #     gt_sentence = gt_sentence.expand_as(sentences) # [b,c]
        #     sim = torch.cosine_similarity(gt_sentence,sentences,dim=1) # [b,c] -> [b]
        #     idx = torch.argsort(sim,descending=True) # 降序

        #     if len(idx) > 11:
        #         neg_sentences = sentences[idx[6:-5]]
        #     else:
        #         neg_sentences = sentences[idx[1:]]
            
        #     pos_score = torch.cosine_similarity(video_postive,sentences[i],dim=0) 
        #     temp = video_postive.expand_as(neg_sentences)
        #     neg_score = torch.cosine_similarity(temp,neg_sentences,dim=1) # [n,c] -> [n]
        #     pos_score = pos_score.expand_as(neg_score)
        #     vas = (opt.global_margin + neg_score - pos_score).clamp(min=0) # [b]
        #     vas_loss.append(vas.mean())

        #     if sim[idx[1]] > 0.9:
        #         pse_video = videos[idx[1]].view(c,-1) # [c,d*t]
        #         temp = video_postive.unsqueeze(1).expand_as(pse_video)
        #         vsim = torch.cosine_similarity(temp,pse_video,dim=0) #[d*t]
        #         vsim = F.softmax(vsim)
        #         p_v = torch.sum(vsim*pse_video,dim=1) # [c,d*t] -> [c]
        #         n_v = torch.sum((1-vsim)*mask.view(-1).float()*pse_video,dim=-1)/(valid_num-1)
        #         p_s = torch.cosine_similarity(video_postive,p_v,dim=0)
        #         n_s = torch.cosine_similarity(video_postive,n_v,dim=0)
        #         crov = (opt.global_margin + n_s - p_s).clamp(min=0) # [1]
        #         crov_loss.append(crov)
        #     else:
        #         crov_loss.append(torch.tensor(0.).cuda())


        for i in range(b):
            # sentence1 = sentences1[i] # [c]
            # sentence1 = sentence1.view(1,c//4,1,1).repeat(b,1,d,t) # [c] -> [b,c,d,t]
            # score1 = torch.cosine_similarity(videos1, sentence1, dim=1).squeeze(1) # [b,1,d,t] -> [b,d,t]
            
            # # score1 = norm(score1)
            # # plot_map(score1,'score1')
            
            # sentence2 = sentences2[i] # [c]
            # sentence2 = sentence2.view(1,c//4,1,1).repeat(b,1,d,t) # [c] -> [b,c,d,t]
            # score2 = torch.cosine_similarity(videos2, sentence2, dim=1).squeeze(1) # [b,1,d,t] -> [b,d,t]
            
            # # score2 = norm(score2)
            # # plot_map(score2,'score2')
            
            # sentence3 = sentences3[i] # [c]
            # sentence3 = sentence3.view(1,c//4,1,1).repeat(b,1,d,t) # [c] -> [b,c,d,t]
            # score3 = torch.cosine_similarity(videos3, sentence3, dim=1).squeeze(1) # [b,1,d,t] -> [b,d,t]
            
            # # score3 = norm(score3)
            # # plot_map(score3,'score3')
            
            # sentence4 = sentences4[i] # [c]
            # sentence4 = sentence4.view(1,c//4,1,1).repeat(b,1,d,t) # [c] -> [b,c,d,t]
            # score4 = torch.cosine_similarity(videos4, sentence4, dim=1).squeeze(1) # [b,1,d,t] -> [b,d,t]
            
            # # score4 = norm(score4)
            # # plot_map(score4,'score4')
            # # exit()
            # score = (score1+score2+score3+score4)/4

            sentence = sentences[i] # [c]
            sentence = sentence.view(1,c,1,1).repeat(b,1,d,t) # [c] -> [b,c,d,t]
            score = torch.cosine_similarity(videos, sentence, dim=1).squeeze(1) # [b,1,d,t] -> [b,d,t]

            #sentence = sentence.permute(0,2,3,1) # [b,c,d,t] -> [b,d,t,c]
            
            #score = F.sigmoid(self.fc(torch.cat([videos,sentence],dim=-1))).squeeze()

            # norm:start 
            # score = score.view(b,-1)
            # score_min = torch.min(score, dim = -1, keepdim = True)[0]
            # score_max = torch.max(score, dim = -1, keepdim = True)[0]
            # score_norm = (score - score_min)/(score_max-score_min)
            # score = score - score_min
            # score = score*score_norm
            # score = score.view(b,d,t)
            # norm:end
            
            score = score.masked_fill(mask == 0, float('-inf'))
            postive_map.append(score[i]) #[d,t]

            score = score.view(b,-1)
            orders = torch.argsort(score, dim=1, descending=True)[:,:valid_num].detach().cpu().numpy()

            # diversity_loss:start
            # temp = F.softmax(score[i],dim=-1) #*opt.lambda_softmax #[1,d*t]
            # diversity = -torch.sum(temp*torch.log(temp+1e-8)) 
            # diversity_loss.append(diversity)
            # diversity_loss:end

            # fb_loss_postive:start
            # videos = videos.permute(0,3,1,2)
            # video = videos[i].view(c,d*t)
            # postive = torch.sum(temp*video,dim=-1) # [c,d*t]
            # negtive = torch.sum((1-temp)*mask.view(-1).float()*video,dim=-1)/(valid_num-1)
            # postive_score = torch.cosine_similarity(postive, sentences[i],dim=0)
            # negative_score = torch.cosine_similarity(negtive, sentences[i],dim=0)
            # fb = (opt.global_margin + negative_score - postive_score).clamp(min=0)
            # fb_loss.append(fb)
            # fb_loss_postive:end

            # fb_loss_all:start
            # temp = F.softmax(score,dim=-1).unsqueeze(1) # [b,1,d*t]
            # video_tmp = videos.view(b,c,d*t) # [b,c,d*t]
            # postive = torch.sum(temp*video_tmp,dim=-1) # [b,c,d*t] -> [b,c]
            # negtive = torch.sum((1-temp)*mask.view(1,1,d*t).float()*video_tmp,dim=-1)/(valid_num-1)
            # postive_score = torch.cosine_similarity(postive, sentences,dim=1) # [b,c] & [b,c] -> [b]
            # negative_score = torch.cosine_similarity(negtive, sentences,dim=1)
            # fb = (opt.global_margin + negative_score - postive_score).clamp(min=0) # [b]
            # fb_loss.append(fb.mean())
            # fb_loss_all:end

            # vas_loss:start


            # vas_loss:end
            

            # score = LogSumExp(score, opt.lambda_lse, dim=-1) # [b,d*t] -> [b]
            score = get_video_score_nms(score, lam, iou_maps, orders)
            score_map.append(score)

        postive_map = torch.stack(postive_map) # [b,d,t]
        scores = torch.stack(score_map) # [b,b]

        # diversity_loss = torch.stack(diversity_loss)

        # fb_loss = torch.stack(fb_loss).sum() # [b]
        # fb_loss = torch.log(1+torch.sum(torch.exp(0.1*fb_loss)))
        # fb_loss = LogSumExp(score, opt.lambda_lse, dim=-1)

        #vas_loss = torch.stack(vas_loss).mean()

        #crov_loss = torch.stack(crov_loss).mean()

        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (opt.global_margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (opt.global_margin + scores - d2).clamp(min=0)

        # clear diagonals
        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if opt.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        
        if is_training and iters % opt.log_step == 0:
            writer.add_scalar('cost_s',cost_s.sum()/b,iters)
            writer.add_scalar('cost_im',cost_im.sum()/b,iters)
            #writer.add_scalar('fb_loss',fb_loss,iters)
            #writer.add_scalar('crov_loss',crov_loss,iters)
            #writer.add_scalar('lam',lam,iters)

        return cost_s.sum()/b + cost_im.sum()/b, postive_map


def Criterion(videos, sentences, opt, writer, iters, is_training, lam, mask, match_map, valid_num, iou_maps):

    # videos[b,c,d,t] sentences[b,c]
    b,c,d,t = videos.size()
    postive_map = []
    score_map = []
    cmil_loss = []
    postive_loss = []
    negative_loss = []

    for i in range(b):
        sentence = sentences[i] # [c]
        global_sentence = sentence.view(1,c).repeat(b,1)
        sentence = sentence.view(1,c,1,1).repeat(b,1,d,t) # [c] -> [b,c,d,t]
        score = torch.cosine_similarity(videos, sentence, dim=1).squeeze(1) # [b,1,d,t] -> [b,d,t]
        # global_score = torch.cosine_similarity(global_sentence, global_video, dim=1)

        # score = score.view(b,-1)
        # score_min = torch.min(score, dim = -1, keepdim = True)[0]
        # score_max = torch.max(score, dim = -1, keepdim = True)[0]
        # score_norm = (score - score_min)/(score_max-score_min)
        # score = score - score_min
        # score = score*score_norm
        # score = score.view(b,d,t)
        
        score = score.masked_fill(mask == 0, float('-inf'))
        postive_map.append(score[i]) #[d,t]
        plot_map(score,'hold_video')
        exit()
        score = score.view(b,-1)
        orders = torch.argsort(score, dim=1, descending=True)[:,:valid_num].detach().cpu().numpy()
        #score = F.softmax(score*opt.lambda_softmax,dim=-1)
        
        #score = LogSumExp(score, opt.lambda_lse, dim=-1) # [b,d*t] -> [b]

        score,neg_score,p_loss,n_loss = get_video_score_nms(score, lam, iou_maps, orders)


        postive_loss.append(p_loss[i])
        negative_loss.append(n_loss[i])
        cmil_loss.append((opt.global_margin + neg_score[i] - score[i]).clamp(min=0).mean())
        # score = score + global_score    
        score_map.append(score)

    postive_loss = torch.stack(postive_loss).mean()
    negative_loss = torch.stack(negative_loss).mean()
    cmil_loss = torch.stack(cmil_loss).mean()
    postive_map = torch.stack(postive_map) # [b,d,t]
    scores = torch.stack(score_map) # [b,b]

    diagonal = scores.diag().view(scores.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    # compare every diagonal score to scores in its column
    # caption retrieval
    cost_s = (opt.global_margin + scores - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (opt.global_margin + scores - d2).clamp(min=0)

    # clear diagonals
    I = torch.eye(scores.size(0)) > .5
    if torch.cuda.is_available():
        I = I.cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)

    # keep the maximum violating negative for each query
    if opt.max_violation:
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]
    
    if is_training and iters % opt.log_step == 0:
        writer.add_scalar('cost_s',cost_s.sum()/b,iters)
        writer.add_scalar('cost_im',cost_im.sum()/b,iters)
        writer.add_scalar('cmil_loss',cmil_loss,iters)
        writer.add_scalar('postive_loss',postive_loss,iters)
        writer.add_scalar('negative_loss',negative_loss,iters)

    return cost_s.sum()/b + cost_im.sum()/b + cmil_loss + postive_loss + negative_loss, postive_map


def Criterion1(videos, sentences, opt, writer, iters, is_training, max_iter, mask, match_map, valid_num):

    # videos[b,c,d,t] sentences[b,c]
    b,c,d,t = videos.size()
    postive_map = []
    score_map = []
    #lam = get_lambda(iters, max_iter, opt.continuation_func)

    for i in range(b):
        sentence = sentences[i] # [c]
        sentence = sentence.view(1,c,1,1).repeat(b,1,d,t) # [c] -> [b,c,d,t]
        score = torch.cosine_similarity(videos, sentence, dim=1).squeeze(1) # [b,1,d,t] -> [b,d,t]


        score = score.view(b,-1)
        score_min = torch.min(score, dim = -1, keepdim = True)[0]
        score_max = torch.max(score, dim = -1, keepdim = True)[0]
        score_norm = (score - score_min)/(score_max-score_min)
        score = score - score_min
        score = score*score_norm
        score = score.view(b,d,t)
        
        score = score.masked_fill(mask.squeeze() == 0, float('-inf'))
        postive_map.append(score[i]) #[d,t]

        score = score.view(b,-1)
        #orders = torch.argsort(score, dim=1, descending=True)[:,:valid_num].detach().cpu().numpy()
        #postive_orders.append(orders[i])
        #score = F.softmax(score*opt.lambda_softmax,dim=-1)
        
        score = LogSumExp(score, opt.lambda_lse, dim=-1) # [b,d*t] -> [b]

        #score = get_video_score_nms(score, lam, opt.temporal_scale, match_map, orders)
            
        score_map.append(score)

    postive_map = torch.stack(postive_map) # [b,d,t]
    scores = torch.stack(score_map) # [b,b]

    diagonal = scores.diag().view(scores.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    # compare every diagonal score to scores in its column
    # caption retrieval
    cost_s = (opt.global_margin + scores - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (opt.global_margin + scores - d2).clamp(min=0)

    # clear diagonals
    I = torch.eye(scores.size(0)) > .5
    if torch.cuda.is_available():
        I = I.cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)

    # keep the maximum violating negative for each query
    if opt.max_violation:
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

    # if is_training and iters % opt.log_step == 0:
    #     writer.add_scalar('cost_s',cost_s.sum()/b,iters)
    #     writer.add_scalar('cost_im',cost_im.sum()/b,iters)
    #     #writer.add_scalar('L1_loss',L1_loss,iters)
    #     #writer.add_scalar('lam',lam,iters)
    return cost_s.sum()/b + cost_im.sum()/b, postive_map #+ 0.0001*L1_loss

def Caption_Criterion(pred, word_id, sentence_lengths):
    word_id = word_id.transpose(0,1).cuda()
    target_captions = pack_padded_sequence(word_id,sentence_lengths)[0]
    caption_loss = F.cross_entropy(pred,target_captions)

    return caption_loss

def pem_cls_loss_func(pred_score, gt_iou_map, mask):
    gt_iou_map = gt_iou_map * mask
    pmask = (gt_iou_map > 0.9).float()
    nmask = (gt_iou_map <= 0.9).float()
    nmask = nmask * mask

    num_positive = torch.sum(pmask)
    num_entries = num_positive + torch.sum(nmask)
    ratio = num_entries / num_positive
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    epsilon = 0.000001
    loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
    loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * nmask
    loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
    return loss