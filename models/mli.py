import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLI(nn.Module):
    """
        Multi-modality Latent Interaction module (MLI)
    """
    def __init__(self,opt):
        super(MLI,self).__init__()

        self.opt = opt
        self.video_linear = nn.Linear(opt.joint_dim,opt.num_sets)
        self.sentence_linear = nn.Linear(opt.joint_dim,opt.num_sets)
        self.interaction = nn.Linear(opt.joint_dim,opt.joint_dim)
        self.cross_dim = nn.Linear(opt.joint_dim,opt.joint_dim)
        self.cross_channel = nn.Linear(opt.num_sets**2,opt.num_sets**2)
        self.Qv = nn.Linear(opt.join_dim,128)
        self.Qs = nn.Linear(opt.join_dim,128)
        self.K = nn.Linear(opt.join_dim,128)
        self.V = nn.Linear(opt.join_dim,128)
    
    def forward(self,videos,videos_mask,sentences,sentences_mask):

        weights = self.video_linear(videos).transpose(1,2)
        mask = videos_mask.unsqueeze(1)
        mask = mask.expand_as(weights)
        masked_videos = videos.masked_fill(mask == True, float('-inf'))
        weights = F.softmax(masked_videos,dim=-1)
        videos_latent = torch.bmm(weights,videos)

        weights = self.sentence_linear(sentences).transpose(1,2)
        mask = sentences_mask.unsqueeze(1)
        mask = mask.expand_as(weights)
        masked_sentences = sentences.masked_fill(mask == True, float('-inf'))
        weights = F.softmax(masked_sentences,dim=-1)
        sentences_latent = torch.bmm(weights,sentences)

        # interaction [B,K,K,D]->[B,K*K,D]
        interaction = videos_latent.unsqueeze(2)*sentences_latent.unsqueeze(1)
        size = interaction.size()
        interaction = self.interaction(interaction).view(size[0],-1,size[-1])

        # latent [B,K*K,D]
        cross_dim = self.cross_dim(interaction)
        cross_channel = self.cross_channel(interaction.transpose(1,2)).transpose(1,2)
        latent = cross_dim+cross_channel

        # videos_query,sentences_query [B,N,128]
        videos_query = self.Qv(videos)
        sentences_query = self.Qs(sentences)

        # key,value [B,K*K,128]
        key = self.K(latent)
        value = self.V(latent)
        dim = key.size(-1)

        # [B,N,K*K]
        videos_attention = F.softmax(torch.bmm(videos_query,key.transpose(1,2))/math.sqrt(dim),dim=-1)
        sentences_attention = F.softmax(torch.bmm(sentences_query,key.transpose(1,2))/math.sqrt(dim),dim=-1)

        updated_videos = videos + torch.bmm(videos_attention,latent)
        updated_sentences = sentences + torch.bmm(sentences_attention,latent)

        return updated_videos,updated_sentences
