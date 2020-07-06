# coding:utf8
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from tools.beam_search import CaptionGenerator
import torch.nn.functional as F
  
class VideoCaption(nn.Module):
    def __init__(self, opt, vocab):
        super(VideoCaption, self).__init__()
        self.word2idx = vocab.word2idx
        self.opt = opt
        
        self.rnn = nn.GRU(opt.joint_dim, opt.joint_dim, num_layers=opt.RNN_layers)
        nn.init.orthogonal_(self.rnn.weight_ih_l0)     
        nn.init.orthogonal_(self.rnn.weight_hh_l0)
        
        self.classifier = nn.Linear(opt.joint_dim, len(self.word2idx))
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, video_embedding, caption_word_embedding, caption_length):
        
        b,c,d,t = video_embedding.size()
        video_embedding = video_embedding.view(b,c,-1)
        video_embedding = torch.sum(video_embedding,dim=-1).unsqueeze(0) #[1,b,c]

        caption_word_embedding = caption_word_embedding.transpose(0,1) #[l,b,c]
        video_embedding = video_embedding.expand_as(caption_word_embedding)

        packed_embeddings = pack_padded_sequence(video_embedding, caption_length)
        outputs, state = self.rnn(packed_embeddings)

        pred = self.classifier(outputs[0])
        return pred

