import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tools.util import PositionalEncoding,multihead_mask,l2norm
from models.graph_convolution import GraphConvolution
from transformers import BertModel, BertConfig

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
        nn.init.orthogonal_(self.rnn.weight_ih_l0)     
        nn.init.orthogonal_(self.rnn.weight_hh_l0)
        nn.init.orthogonal_(self.rnn.weight_ih_l0_reverse)
        nn.init.orthogonal_(self.rnn.weight_hh_l0_reverse)  

    def forward(self, sentences, sentence_lengths):
        """Handles variable size captions
        """
        sentences = sentences.transpose(0,1) # [b,l,c]->[l,b,c]
        mask = multihead_mask(sentences, sentence_lengths)
        packed = pack_padded_sequence(sentences, sentence_lengths)
        out, _ = self.rnn(packed)
        padded = pad_packed_sequence(out)

        sentences = padded[0].transpose(0,1) # [l,b,c]->[b,l,c]

        I = torch.LongTensor(sentence_lengths).view(-1, 1, 1) # view的作用类似reshape
        I = I.expand(sentences.size(0), 1, self.opt.joint_dim)-1
        I = I.cuda()
        global_sentences = torch.gather(sentences, 1, I).squeeze(1)
        # normalization in the joint embedding space
        global_sentences = l2norm(global_sentences,dim=-1)
        
        return global_sentences,sentences,mask

class SentenceEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.max_num_words = args.max_num_words
        self.gcn_layers = nn.ModuleList([
            GraphConvolution(args.word_dim)
            for _ in range(args.num_gcn_layers)
        ])

    def forward(self, x, mask, node_pos, node_mask, adj_mat):
        length = mask.sum(dim=-1)

        for g in self.gcn_layers:
            res = x
            x = g(x, node_mask, adj_mat)
            x = F.dropout(x, self.dropout, self.training)
            x = res + x

        x = F.dropout(x, self.dropout, self.training)

        return x

class TextNet(nn.Module):
    def __init__(self, opt): #code_length为fc映射到的维度大小
        super(TextNet, self).__init__()

        modelConfig = BertConfig.from_pretrained('/home/share/wangyunxiao/BERT/bert-base-uncased-config.json')
        self.textExtractor = BertModel.from_pretrained('/home/share/wangyunxiao/BERT/bert-base-uncased-pytorch_model.bin', config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size
        for param in self.textExtractor.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(embedding_dim, opt.joint_dim)

    def forward(self, tokens, segments, input_masks):
        self.textExtractor.eval()
        output=self.textExtractor(tokens, token_type_ids=segments,
                                 		attention_mask=input_masks)
        text_embeddings = output[0]
        #output[0](batch size, sequence length, model hidden dimension)

        features = self.fc(text_embeddings)
        features = torch.tanh(features)
        return features
