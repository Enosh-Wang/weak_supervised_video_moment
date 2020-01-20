import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.util import PositionalEncoding,multihead_mask,l2norm

class TextEncoder(nn.Module):

    def __init__(self, opt):
        super(TextEncoder, self).__init__() 

        self.opt = opt
        self.PE = PositionalEncoding(d_model=opt.word_dim,dropout=opt.dropout,max_len=1000)
        self.attention = nn.ModuleList([nn.MultiheadAttention(embed_dim=opt.word_dim,num_heads=opt.sentence_heads) for _ in range(opt.sentence_attn_layers)])
        #self.rnn = nn.GRU(opt.word_dim, opt.joint_dim//2, opt.RNN_layers, bidirectional=True)
        self.fc = nn.Linear(opt.word_dim,opt.joint_dim)

    def forward(self, sentences, sentence_lengths):
        """Handles variable size captions
        """
        sentences = sentences.transpose(0,1)

        mask = multihead_mask(sentences, sentence_lengths)
        sentences = self.PE(sentences)

        for layer in self.attention:
            res = sentences
            sentences, _ = layer(sentences,sentences,sentences,mask)
            sentences = F.dropout(sentences,self.opt.dropout,self.training)
            sentences = sentences + res


        '''
        packed = pack_padded_sequence(sentences, sentence_lengths)
        out, _ = self.rnn(packed)
        padded = pad_packed_sequence(out)
        
        sentences = padded[0].transpose(0,1)

        I = torch.LongTensor(sentence_lengths).view(-1, 1, 1) # view的作用类似reshape
        I = I.expand(sentences.size(0), 1, self.opt.joint_dim)-1
        I = I.cuda()
        sentences = torch.gather(sentences, 1, I).squeeze(1)
        # normalization in the joint embedding space
        sentences = l2norm(sentences)
        '''
        sentences = sentences.transpose(0,1)
        sentences = self.fc(sentences)
        return sentences, mask

