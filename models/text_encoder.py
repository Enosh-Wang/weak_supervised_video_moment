import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tools.util import PositionalEncoding,multihead_mask,l2norm
from graph_convolution import GraphConvolution

class TextEncoderMulti(nn.Module):

    def __init__(self, opt):
        super(TextEncoderMulti, self).__init__() 

        self.opt = opt
        self.PE = PositionalEncoding(d_model=opt.joint_dim,dropout=opt.dropout,max_len=1000)
        self.attention = nn.ModuleList([nn.MultiheadAttention(embed_dim=opt.joint_dim,num_heads=opt.sentence_heads) for _ in range(opt.sentence_attn_layers)])
        self.fc = nn.Linear(opt.word_dim,opt.joint_dim)

    def forward(self, sentences, sentence_lengths):
        """Handles variable size captions
        """
        sentences = self.fc(sentences)
        sentences = sentences.transpose(0,1)

        mask = multihead_mask(sentences, sentence_lengths)
        sentences = self.PE(sentences)

        for layer in self.attention:
            res = sentences
            sentences, _ = layer(sentences,sentences,sentences,mask)
            sentences = F.dropout(sentences,self.opt.dropout,self.training)
            sentences = sentences + res

        sentences = sentences.transpose(0,1)
        
        return sentences, mask

class TextEncoderGRU(nn.Module):

    def __init__(self, opt):
        super(TextEncoderGRU, self).__init__() 

        self.opt = opt
        self.rnn = nn.GRU(opt.word_dim, opt.joint_dim//2, opt.RNN_layers, bidirectional=True)
        self.norm = nn.BatchNorm1d(opt.joint_dim)

    def forward(self, sentences, sentence_lengths):
        """Handles variable size captions
        """
        sentences = sentences.transpose(0,1)

        packed = pack_padded_sequence(sentences, sentence_lengths)
        out, _ = self.rnn(packed)
        padded = pad_packed_sequence(out)

        #sentences = self.norm(padded[0]).transpose(0,1)
        # [l,b,c] -> [b,c,l]
        sentences = padded[0].permute(1,2,0)
        sentences = self.norm(sentences).transpose(1,2)
        I = torch.LongTensor(sentence_lengths).view(-1, 1, 1) # view的作用类似reshape
        I = I.expand(sentences.size(0), 1, self.opt.joint_dim)-1
        I = I.cuda()
        sentences = torch.gather(sentences, 1, I).squeeze(1)
        # normalization in the joint embedding space
        #sentences = l2norm(sentences)
        #sentences = sentences.transpose(0,1)

        return sentences

class SentenceEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.max_num_words = args.max_num_words
        self.gcn_layers = nn.ModuleList([
            GraphConvolution(args.word_dim)
            for _ in range(args.num_gcn_layers)
        ])
        self.rnn = DynamicGRU(args.word_dim, args.d_model >> 1, bidirectional=True, batch_first=True)

    def forward(self, x, mask, node_pos, node_mask, adj_mat):
        length = mask.sum(dim=-1)

        # graph_input = torch.cat([
        #     torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(x, node_pos)
        #     # [1, num_nodes, embed_dim]
        # ], 0)

        # x = graph_input
        for g in self.gcn_layers:
            res = x
            x = g(x, node_mask, adj_mat)
            x = F.dropout(x, self.dropout, self.training)
            x = res + x

        x = self.rnn(x, length, self.max_num_words)
        x = F.dropout(x, self.dropout, self.training)

        return x