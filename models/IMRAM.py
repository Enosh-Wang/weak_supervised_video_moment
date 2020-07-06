import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from models.CMIL import get_lambda, get_video_score_nms
from tools.util import LogSumExp, l1norm, l2norm

def frame_by_word(query, query_mask, context, context_mask, opt, eps=1e-8, weight=None):
    """
    query: (n_context, queryL, d) 
    context: (n_context, sourceL, d) 
    """
    smooth = opt.lambda_softmax
    batch_size = context.size(0)
    sourceL = context.size(1)
    queryL = query.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL) frame-by-word 矩阵
    attn = torch.bmm(context, queryT)

    # 对 frame-by-word 矩阵进行 norm
    if opt.raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        if query_mask is not None:
            attn = attn.masked_fill(query_mask == 0,float('-inf'))
        attn = F.softmax(attn, dim=1)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.raw_feature_norm == "l2norm":
        if query_mask is not None:
            attn = attn.masked_fill(query_mask.unsqueeze(1) == 0, 0)
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped":
        if query_mask is not None:
            attn = attn.masked_fill(query_mask.unsqueeze(1) == 0, 0)
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    if context_mask is not None:
        attn = attn.masked_fill(context_mask.unsqueeze(1) == 0,float('-inf'))
        #attn = attn.masked_fill(query_mask.unsqueeze(2) == 0,float('-inf'))
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)

    attn = F.softmax(attn*smooth, dim=1)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext

def func_attention_text(query, video_mask, context, opt, smooth, eps=1e-8, weight=None):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)


    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    if opt.raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = F.softmax(attn, dim=1)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "l1norm":
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)
    
    if weight is not None:
      attn = attn + weight

    video_mask = video_mask.unsqueeze(1)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous().masked_fill(video_mask == True,float('-inf'))
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)

    attn = F.softmax(attn*smooth, dim=1)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT

def func_attention_image(query, video_mask, context, opt, smooth, eps=1e-8, weight=None):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)


    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    if opt.raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = F.softmax(attn, dim=1)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "l1norm":
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)
    
    if weight is not None:
      attn = attn + weight

    #video_mask = video_mask.unsqueeze(1)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()#.masked_fill(video_mask == True,float('-inf'))
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)

    attn = F.softmax(attn*smooth, dim=1)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = opt.global_margin
        self.max_violation = opt.max_violation

    def forward(self, scores):

        diagonal = scores.diag().view(-1, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()

class SCAN(nn.Module):
    """
    Stacked Cross Attention Network (SCAN) model
    """
    def __init__(self, opt):
        super(SCAN, self).__init__()
        # Build Models
        self.opt = opt
        
        print("*********using gate to fusion information**************")
        self.linear_t2i = nn.Linear(opt.joint_dim * 2, opt.joint_dim)
        self.gate_t2i = nn.Linear(opt.joint_dim * 2, opt.joint_dim)
        self.linear_i2t = nn.Linear(opt.joint_dim * 2, opt.joint_dim)
        self.gate_i2t = nn.Linear(opt.joint_dim * 2, opt.joint_dim)

    def gated_memory_t2i(self, input_0, input_1):

        input_cat = torch.cat([input_0, input_1], 2)
        input_1 = torch.tanh(self.linear_t2i(input_cat))
        gate = torch.sigmoid(self.gate_t2i(input_cat))
        output = input_0 * gate + input_1 * (1 - gate)

        return output
    
    def gated_memory_i2t(self, input_0, input_1):

        input_cat = torch.cat([input_0, input_1], 2)
        #input_1 = torch.tanh(self.linear_i2t(input_cat))
        #gate = torch.sigmoid(self.gate_i2t(input_cat))
        #output = input_0 * gate + input_1 * (1 - gate)
        
        output = self.linear_i2t(input_cat)

        return output

    def forward_score(self, img_emb, video_mask, cap_emb, cap_len,iters, max_iter,valid_num, match_map, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        if self.opt.model_mode == "text_IMRAM":
            scores_t2i, attn = self.xattn_score_Text_IMRAM(img_emb, video_mask, cap_emb, cap_len, self.opt)
            scores_t2i = torch.stack(scores_t2i, 0).sum(0)
            score = scores_t2i
        elif self.opt.model_mode == "image_IMRAM":
            scores_i2t, attn = self.xattn_score_Image_IMRAM(img_emb, video_mask, cap_emb, cap_len, self.opt,iters, max_iter,valid_num, match_map)
            scores_i2t = torch.stack(scores_i2t, 0).sum(0)
            score = scores_i2t
        
        return score, attn
    
    def xattn_score_Text_IMRAM(self, images, video_mask, captions_all, cap_lens, opt):
        """
        Images: (n_image, n_regions, d) matrix of images
        captions_all: (n_caption, max_n_word, d) matrix of captions
        CapLens: (n_caption) array of caption lengths
        """
        similarities = [[] for _ in range(opt.iteration_step)]
        
        n_image, n_channel, d, t = images.size()
        images = images.permute(0,2,3,1).view(n_image, -1, n_channel) # ->[b,d,t,c]->[b,d*t,c]
        
        n_caption = captions_all.size(0)

        all_attn = []
        all_postive = torch.zeros_like(captions_all).cuda()
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions_all[i, :n_word, :].unsqueeze(0).contiguous()
            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            
            query = cap_i_expand
            context = images
            attn_iter = []
            weight = [0.6,0.2,0.2]
            for j in range(opt.iteration_step):
                # "feature_update" by default:
                # attn_feat [b,l,c] attn [b,t,l]
                attn_feat, attn = func_attention_text(query, video_mask, context, opt, smooth=opt.lambda_softmax)
                attn = attn.mean(dim=2)
                attn_iter.append(attn*weight[j])

                row_sim = torch.cosine_similarity(cap_i_expand, attn_feat, dim=2)
                row_sim = row_sim.mean(dim=1, keepdim=True)
                similarities[j].append(row_sim)

                query = self.gated_memory_t2i(query, attn_feat)

            attn_iter = torch.stack(attn_iter).sum(dim=0)
            all_attn.append(attn_iter)
            all_postive[i,:n_word] = attn_feat[i]
        # (n_image, n_caption)
        new_similarities = []
        for j in range(opt.iteration_step):
            if len(similarities[j]) == 0:
                new_similarities.append([])
                continue
            similarities_one = torch.cat(similarities[j], 1).double()
            if self.training:
                similarities_one = similarities_one.transpose(0,1)
            new_similarities.append(similarities_one)
        
        return new_similarities,torch.stack(all_attn)#,all_postive

    def xattn_score_Image_IMRAM(self, images, video_mask, captions_all, cap_lens, opt,iters, max_iter,valid_num, match_map):
        """
        Images: (batch_size, n_regions, d) matrix of images
        captions_all: (batch_size, max_n_words, d) matrix of captions
        CapLens: (batch_size) array of caption lengths
        """
        similarities = [[] for _ in range(opt.iteration_step)]
        
        n_image, n_channel, d, t = images.size()
        images = images.permute(0,2,3,1).view(n_image, -1, n_channel)
        
        n_caption = captions_all.size(0)

        all_sim = []

        lam = get_lambda(iters, max_iter, opt.continuation_func)

        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions_all[i, :n_word, :].unsqueeze(0).contiguous()
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            
            query = images
            context = cap_i_expand
            sim_out = []
            for j in range(opt.iteration_step):
                # attn_feat [b,t,c] attn [b,l,t] attn_feat 聚合后的文本特征
                attn_feat, _ = func_attention_image(query, video_mask, context, opt, smooth=opt.lambda_softmax)

                row_sim = torch.cosine_similarity(images, attn_feat, dim=2).masked_fill(video_mask == 0,float('-inf')) # -> [b,t]
                #row_sim = F.softmax(row_sim,dim=1)
                sim_out.append(row_sim[i])
                row_sim = LogSumExp(row_sim, opt.lambda_lse, dim=1, keepdim=True)
                #row_sim = row_sim.max(dim=1, keepdim=True)
                #orders = torch.argsort(row_sim, dim=1, descending=True)[:,:valid_num].detach().cpu().numpy()
                #row_sim = get_video_score_nms(row_sim, lam, opt.temporal_scale, match_map, orders).unsqueeze(1)

                similarities[j].append(row_sim)

                query = self.gated_memory_i2t(query, attn_feat)
            
            sim_out = torch.stack(sim_out).sum(0)
            #sim_out = torch.cosine_similarity(images,query,dim=2)
            #sim_out = sim_out.masked_fill(video_mask == True,float('-inf'))
            #sim_out = F.softmax(sim_out,dim=1)
            all_sim.append(sim_out)
        # (n_image, n_caption)
        new_similarities = []
        for j in range(opt.iteration_step):
            if len(similarities[j]) == 0:
                new_similarities.append([])
                continue
            similarities_one = torch.cat(similarities[j], 1).double()
            if self.training:
                similarities_one = similarities_one.transpose(0,1)
            new_similarities.append(similarities_one)

        return new_similarities,torch.stack(all_sim).view(n_image,d,t)
