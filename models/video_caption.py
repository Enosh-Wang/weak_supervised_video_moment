# coding:utf8
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from tools.beam_search import CaptionGenerator


class VideoCaption(nn.Module):
    def __init__(self, joint_dim, word_dim, RNN_layers, vocab):
        super(VideoCaption, self).__init__()
        self.idx2word = vocab.idx2word
        self.word2idx = vocab.word2idx
        self.fc = nn.Linear(joint_dim, word_dim) # 将视频帧映射到和词向量相同的维度 default=300

        self.rnn = nn.GRU(word_dim, joint_dim//2, num_layers=RNN_layers)
        self.classifier = nn.Linear(joint_dim//2, len(self.word2idx))

    def forward(self, video_embedding, caption_word_embedding, caption_length):
        # img_feats是1024维的向量,通过全连接层转为300维的向量,和词向量一样

        video_embedding = self.fc(video_embedding).unsqueeze(0)
        # 将img_feats看成第一个词的词向量 
        caption_word_embedding = caption_word_embedding.transpose(0,1)
        caption_word_embedding = torch.cat([video_embedding, caption_word_embedding], 0)
        # PackedSequence
        packed_embeddings = pack_padded_sequence(caption_word_embedding, caption_length)
        outputs, state = self.rnn(packed_embeddings)
        # GRU的输出作为特征用来分类预测下一个词的序号
        # 因为输入是PackedSequence,所以输出的output也是PackedSequence
        # PackedSequence第一个元素是Variable,第二个元素是batch_sizes,
        # 即batch中每个样本的长度
        pred = self.classifier(outputs[0])
        return pred, state

    def generate(self, img, eos_token='<end>', # '</EOS>',看代码这里应该只在结尾加了标识，没有在开头加，稍微有些不同
                 beam_size=3,
                 max_caption_length=30,
                 length_normalization_factor=0.0):
        """
        根据图片生成描述,主要是使用beam search算法以得到更好的描述
        """
        cap_gen = CaptionGenerator(embedder=self.embedding,
                                   rnn=self.rnn,
                                   classifier=self.classifier,
                                   eos_id=self.word2idx[eos_token],
                                   beam_size=beam_size,
                                   max_caption_length=max_caption_length,
                                   length_normalization_factor=length_normalization_factor)
        # img:图像特征
        img = self.fc(img).unsqueeze(0)
        sentences, score = cap_gen.beam_search(img)
        sentences = [' '.join([self.idx2word[idx] for idx in sent])
                     for sent in sentences]
        return sentences

