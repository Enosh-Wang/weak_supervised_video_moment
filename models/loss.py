import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tools.util import LogSumExp, get_mask
from models.CMIL import get_lambda, get_video_score_nms
import time
import random
from torch.nn.utils.rnn import pack_padded_sequence
def Criterion1(videos, sentences, opt, writer, iters, is_training, max_iter, mask, match_map):

    # videos[b,c,d,t] sentences[b,c]
    b,c,d,t = videos.size()
    postive_map = []
    score_map = []
    L1_loss = torch.Tensor([0]).cuda()
    lam = get_lambda(iters, max_iter, opt.continuation_func)

    for i in range(b):
        sentence = sentences[i] # [c]
        sentence = sentence.view(1,c,1,1).repeat(b,1,d,t) # [c] -> [b,c,d,t]
        score = torch.cosine_similarity(videos, sentence, dim=1).view(b,-1) # [b,1,d,t] -> [b,d*t]
        if is_training and iters % opt.log_step == 0 and i==0:
            writer.add_histogram('score',score,iters)
        score_min = torch.min(score, dim = -1, keepdim = True)[0]
        score_max = torch.max(score, dim = -1, keepdim = True)[0]
        score_norm = (score - score_min)/(score_max-score_min)
        
        score = score - score_min
        score = score*score_norm
        if is_training and iters % opt.log_step == 0 and i==0:
            writer.add_histogram('score_norm',score,iters)
        L1_loss += torch.abs(score).sum()
        score = score.masked_fill(mask == 0, float('-inf'))
        
        postive_map.append(score[i].view(d,t)) #[d,t]
        '''
        start = time.time()
        score1 = LogSumExp(score, opt.lambda_lse, dim=-1) # [b,d*t] -> [b]
        end = time.time()
        print(end - start)

        start = time.time()
        score3 = get_video_score(score, lam, opt.temporal_scale, match_map, mask)
        end = time.time()
        print(end - start)
        '''
        start = time.time()
        score = get_video_score_nms(score, lam, opt.temporal_scale, match_map)
        end = time.time()
        print(end - start)
        exit()
        if is_training and iters % opt.log_step == 0 and i==0:
            writer.add_histogram('score_video',score,iters)
            
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

    if is_training and iters % opt.log_step == 0:
        writer.add_scalar('cost_s',cost_s.sum(),iters)
        writer.add_scalar('cost_im',cost_im.sum(),iters)
        writer.add_scalar('L1_loss',L1_loss,iters)
        writer.add_scalar('lam',lam,iters)
    return cost_s.sum() + cost_im.sum(), postive_map #+ 0.0001*L1_loss

def sample1(videos,sentences,video_name):
    
        negative_num = 5
        b,c,d,t = videos.size()
        real_negative = min(negative_num,b)
        new_size = b*real_negative
        new_video = torch.zeros(new_size,c,d,t).cuda()
        new_sentence = torch.zeros(new_size,c).cuda()
        
        for i in range(b):
            x = list(range(b))
            x.remove(i)
            new_sentence[i*real_negative:(i+1)*real_negative] = sentences[i]
            # 至多重复采用5次，防止采样到与gt相同的视频
            for j in range(5):
                y = random.sample(x,real_negative)
                if video_name[i] not in [video_name[k] for k in y]:
                    break
            new_video[i*real_negative:(i+1)*real_negative] = videos[y]
        return new_sentence,new_video,real_negative
def sample(videos,sentences,video_name):
    
        negative_num = 5
        b,c = videos.size()
        real_negative = min(negative_num,b)
        new_size = b*real_negative
        new_video = torch.zeros(new_size,c).cuda()
        new_sentence = torch.zeros(new_size,c).cuda()
        
        for i in range(b):
            x = list(range(b))
            x.remove(i)
            new_sentence[i*real_negative:(i+1)*real_negative] = sentences[i]
            # 至多重复采用5次，防止采样到与gt相同的视频
            for j in range(5):
                y = random.sample(x,real_negative)
                if video_name[i] not in [video_name[k] for k in y]:
                    break
            new_video[i*real_negative:(i+1)*real_negative] = videos[y]
        return new_sentence,new_video,real_negative
def Criterion2(videos, sentences, attn, opt, writer, iters, is_training, max_iter, mask, match_map, video_name,valid_num):

    # videos[b,c,d,t] sentences[b,c]
    b,c,d,t = videos.size()
    videos = videos.view(b,c,-1)
    videos = torch.sum(videos,dim=-1)

    neg_sentence, neg_video, neg_num = sample(videos, sentences, video_name)
    all_video = torch.cat((videos,neg_video),dim=0)
    all_sentence = torch.cat((sentences,neg_sentence),dim=0)
    
    #all_sentence = all_sentence.view(b,c,1,1).repeat(1,1,d,t)
    video_score = torch.cosine_similarity(all_video, all_sentence, dim=1)
    #score = score.masked_fill(mask == 0, float('-inf'))

    #lam = get_lambda(iters, max_iter, opt.continuation_func)

    #video_score = get_video_score_nms(score.view(b,-1), lam, opt.temporal_scale, match_map, valid_num)

    batch_size = videos.size(0)
    positive_score = video_score[:batch_size]
    negative_score = video_score[batch_size:]

    negative_score = negative_score.view(-1,neg_num)
    positive_score = positive_score.unsqueeze(1).expand_as(negative_score)
    global_loss = (opt.global_margin + negative_score - positive_score).clamp(min=0)
    global_loss = global_loss.sum()

    #if is_training and iters % opt.log_step == 0:
        #writer.add_scalar('lam',lam,iters)
        #writer.add_histogram('score',score,iters)
        #writer.add_histogram('video_score',video_score,iters)
    return global_loss, attn#score[:batch_size] #+ 0.0001*L1_loss

def Criterion(pred, word_id, sentence_lengths):
    word_id = word_id.transpose(0,1).cuda()
    target_captions = pack_padded_sequence(word_id,sentence_lengths)[0]
    caption_loss = F.cross_entropy(pred,target_captions)

    return caption_loss
