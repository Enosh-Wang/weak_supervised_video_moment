import torch
import torch.nn as nn
import torch.nn.functional as F
from models.text_encoder import TextEncoder
from models.video_caption import VideoCaption
from models.video_encoder import VideoEncoder
from models.loss import Criterion
from utils.util import cosine_similarity

class Model(nn.Module):

    def __init__(self, opt, vocab):
        super().__init__()
        self.word_embedding = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.word_embedding.weight.data.uniform_(-0.1, 0.1)
        self.video_encoder = VideoEncoder(opt.video_dim, opt.joint_dim)
        self.text_encoder = TextEncoder(opt.vocab_size, opt.word_dim, opt.joint_dim, opt.RNN_layers)
        self.video_caption = VideoCaption(opt.joint_dim, opt.word_dim, opt.RNN_layers, vocab)
    
    def forward(self, videos, sentences, sentence_lengths, video_lengths, margin, is_training):

        word_embedding = self.word_embedding(sentences)
        sentence_embedding = self.text_encoder(word_embedding, sentence_lengths, is_training)
        video_embedding = self.video_encoder(videos, video_lengths)
        similarity_weight = cosine_similarity(video_embedding, sentence_embedding)
        
        # 加mask,计算softmax
        size = similarity_weight.size()
        mask=torch.zeros(size).cuda()
        for i in range(size[0]):
            for j in range(size[1]):
                mask[i,j,video_lengths[i]:] = float('-inf')

        masked_similarity = similarity_weight + mask
        similarity = F.softmax(masked_similarity,dim=2)

        # 提取similarity得分最高的视频片段
        max_arg = torch.max(similarity,2)[1]
        size = video_embedding.size()
        video_max = torch.zeros(size[0],size[2])
        for i in range(size[0]):
            max_arg_diag = max_arg[i,i]
            video_max[i,:] = video_embedding[i,max_arg_diag.data,:]

        video_max = video_max.cuda()
        caption_predict,_ = self.video_caption(video_max, word_embedding, sentence_lengths)
        
        loss = Criterion(similarity,sentences,sentence_lengths,video_lengths,caption_predict,margin)
        
        return video_embedding,sentence_embedding,caption_predict,similarity,loss


