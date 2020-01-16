import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
from utils.util import PositionalEncoding,l2norm

class TextEncoder(nn.Module):

    def __init__(self, vocab_size, word_dim, joint_dim, RNN_layers):

        super(TextEncoder, self).__init__()
        self.joint_dim = joint_dim # default=1024 

        # 加上transformer，尝试通过self-attention区分不同单词的影响程度（不同词性的重要性不同）
        #self.trans = nn.TransformerEncoderLayer(d_model = word_dim,nhead = 1, dropout=0.1,) # nhead要能整除word_dim
        self.rnn = nn.GRU(word_dim, joint_dim//2, RNN_layers, bidirectional=True)
        #self.PE = PositionalEncoding(d_model=word_dim,dropout=0.1,max_len=1000)

    def forward(self, x, lengths, is_training):
        """Handles variable size captions
        """
        # word_embedding word ids to vectors
        seq_size, batch_size, feat_size = x.size()
        src_key_padding_mask = torch.BoolTensor(batch_size,seq_size)
        for index in range(batch_size):
            src_key_padding_mask[index,:lengths[index]] = False
            src_key_padding_mask[index,lengths[index]:] = True
        src_key_padding_mask = src_key_padding_mask.cuda()

        # 维度增加为[seq,batch,feat]
        #x = self.PE(x)
        #x = self.trans(x,src_key_padding_mask=src_key_padding_mask)
        packed = pack_padded_sequence(x, lengths)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out)
        I = torch.LongTensor(lengths).view(-1, 1, 1) # view的作用类似reshape
        I = I.expand(batch_size, 1, self.joint_dim)-1
        I = I.cuda()
        out = padded[0].permute(1,0,2)
        out = torch.gather(out, 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out)

        return out