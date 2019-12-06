import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1).sqrt()
    X = X / norm[:,None]
    return X


def cross_attention(x1, x2, dim=2):
    """Returns cosine similarity based attention between x1 and x2, computed along dim."""
    # 原始代码
    # batch 矩阵相乘 x1[128,26,1024] x2[128,1024] w1[128,26,1]
    # w1=torch.bmm(x1, x2.unsqueeze(2))
    # <!-改为MIL
    batch_size = x1.size()[0]
    x2 = x2.repeat(batch_size,1,1) # [128,128,1024]
    x1 = x1.permute(0,2,1) # [128,1024,14]
    w1 = torch.bmm(x2,x1) # [128,128,14]
    return w1
	
	
class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, use_abs=False,no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
			
        self.ws1 = nn.Linear(img_dim, embed_size)
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Linear(embed_size, embed_size)	

        self.init_weights()

        # 修改
        #self.trans = nn.TransformerEncoderLayer(d_model=img_dim, nhead=1,dim_feedforward=4096)
        #self.self_atten = nn.MultiheadAttention(embed_dim=img_dim,num_heads=1)
        #self.dropout = nn.Dropout(0.1)
        #self.norm = nn.LayerNorm(img_dim)
    def init_weights(self):
        """Xavier initialization for the fully connected layer
           同上
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)
        r1 = np.sqrt(6.) / np.sqrt(self.ws1.in_features +
                                  self.ws1.out_features)
        self.ws1.weight.data.uniform_(-r, r)
        self.ws1.bias.data.fill_(0)
		

    def forward(self, images, cap_embs,lengths_img):
        """Extract image feature vectors."""
        # cap_embs.shape=[128,1024]
        # assuming that the precomputed features are already l2-normalized
        # 前面的作用是将视频特征映射到联合空间
        #size = images.size() # [128, 14, 4096] 

        '''待调试代码区'''
        #images_trans = self.trans(images)
        '''self——attention加shortcut,即transformer里的muli-head和add，norm结构
        '''
        #images_feature = self.self_atten(images,images,images)[0]+images
        #images_feature = self.dropout(images_feature)+images
        #images_feature = self.norm(images_feature)

        image_feature=self.ws1(images) # weight [4096, 1024] feature [128, 14, 1024]
        #size = image_feature.size()
        attn_weights=cross_attention(image_feature, cap_embs, dim=2)
        #attn_weights = self.softmax(attn_weights)
        size = attn_weights.size()
        mask=torch.zeros(size).cuda()
        # 原版attn_weights[128,14,1] 在clip维度上执行softmax
        
        for i in range(size[0]):
            for j in range(size[1]):
                #temp=self.softmax(attn_weights[i,j,0:lengths_img[i]])
                mask[i,j,lengths_img[i]:] = float('-inf')
        # 下面部分代码的作用实现 Text-guided attention，可以结合论文来看
        # attn_weights_s.requires_grad_()
        # attn_weights=attn_weights_s.cuda()
        attn_weights_mask = attn_weights + mask
        attn_weights_soft = self.softmax(attn_weights_mask)
        # img_feature=[128,26,1024] attn_weight=[128,26,1]
        
        # 原版代码
        # out=torch.bmm(image_feature.transpose(1,2),attn_weights)

        # <!-改用max的代码
        #max_index = attn_weights.argmax(dim=1)
        #max_index = torch.squeeze(max_index)
        #out = torch.zeros(size[0],size[2]).cuda()
        #for i in range(size[0]):
        #    out[i] = image_feature[i,max_index[i],:]
        # 改用max的代码-!>

        # <!- 改用MIL的代码
        scores = torch.max(attn_weights_soft,2)[0]
        # 改用MIL的代码 -!>
        '''
        # out [128,1024]
        features = self.fc(out)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding
        if self.use_abs:
            features = torch.abs(features)

        return features, attn_weights
        '''
        return scores,attn_weights_soft

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        # 可训练参数的词典
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        # 加载参数
        super(EncoderImagePrecomp, self).load_state_dict(new_state) # ['ws1.weight', 'ws1.bias', 'fc.weight', 'fc.bias']


# tutorials/08 - Language Model
# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_abs=False):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        # word embedding
        # 转为词向量
        self.embed = nn.Embedding(vocab_size, word_dim)

        # 加上transformer，尝试通过self-attention区分不同单词的影响程度（不同词性的重要性不同）
        #self.trans = nn.TransformerEncoderLayer(d_model = word_dim,nhead = 1) # nhead要能整除word_dim
        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors

        x = self.embed(x)
        # 修改
        #x = self.trans(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1) # view的作用类似reshape
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    # 标准的矩阵乘法 结果[滑窗个数，单词个数]
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).squeeze(2).sqrt().t()
    return score

		
class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self,scores,attn_weights,lengths_img):
        # 取对角线元素 2D -> 1D
        # view的作用类似reshape
        diagonal = scores.diag().view(scores.size(0), 1)
        # 扩展成与scores相同的大小
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        
        # compare every diagonal score to scores in its column
        # caption retrieval

        cost_s = (self.margin + scores - d1).clamp(min=0)

        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        # eye:返回一个2维张量，对角线位置全1，其它位置全0
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        # masked_fill_:在mask值为1的位置处用value填充。mask的元素个数需和本tensor相同，但尺寸可以不同。
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)
        # keep the maximum violating negative for each query
        # 用max代替sum
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        
        # 时间平滑项
        lambda_1 = 8e-5
        size = attn_weights.size()
        cost_smooth = Variable(torch.zeros(size[0],size[2]),requires_grad = True)
        cost_smooth = cost_smooth.cuda()
        for i in range(size[0]):
            a = range(lengths_img[i]-1)
            b = range(1,lengths_img[i])
            cost_smooth[i,:lengths_img[i]-1] = attn_weights[i,i,a]-attn_weights[i,i,b]
        # 时间稀疏项
        lambda_2 = 8e-5
        cost_sparse = Variable(torch.zeros(size[0],size[2]),requires_grad = True)
        cost_sparse = cost_sparse.cuda()
        for i in range(size[0]):
            cost_sparse[i,:lengths_img[i]] = attn_weights[i,i,:lengths_img[i]]
        
        return cost_s.sum() + cost_im.sum() + lambda_1*cost_smooth.pow(2).sum() + lambda_2*cost_sparse.sum()

class VSE(object):
    """
    rkiros/uvs model
    """

    def __init__(self, opt):
        # tutorials/09 - Image Captioning
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImagePrecomp(opt.img_dim, opt.embed_size,
                                    use_abs=opt.use_abs,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_abs=opt.use_abs)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer

        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation)

        # 要优化的参数
        # 文本部分主要是embedding和GRU，图片部分只有一个FC
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        """拼接视频和文本两部分模型的参数列表"""
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        """分别加载视频和文本的两部分参数"""
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    # 删除了volatile
    def forward_emb(self, images, captions, lengths, lengths_img):
        """Compute the image and caption embeddings
        输入依次是一个batch的padding后的视频、padding后的文本，文本的单词数目，视频的滑窗数目
        """
        # Set mini-batch dataset
        # 封装数据类型
        images = Variable(images)
        captions = Variable(captions)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        # 主要进行Embedding和GRU
        cap_init_emb = self.txt_enc(captions, lengths)
        # 主要有两个FC和一个Text-Guide Attention
        # 实验：加入self-attention
        #img_emb, attn_weights = self.img_enc(images,cap_init_emb,lengths_img)
        # [128,128]
        scores, attn_weights = self.img_enc(images,cap_init_emb,lengths_img)
        #cap_emb=cap_init_emb
        #return img_emb, cap_emb, attn_weights
        return  attn_weights, scores

    # 删除了volatile
    def forward_emb_image(self, images):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images)

        if torch.cuda.is_available():
            images = images.cuda()
            #captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)
        return img_emb

    # 删除了volatile
    def forward_emb_caption(self, captions, lengths):
        #"""Compute the image and caption embeddings"""
        # Set mini-batch dataset
        captions = Variable(captions)
        if torch.cuda.is_available():
            captions = captions.cuda()

        # Forward
        cap_emb = self.txt_enc(captions, lengths)
        return cap_emb
    '''
    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb)
        self.logger.update('Le', loss.data, img_emb.size(0))
        return loss
    '''
    def forward_loss(self, scores,attn_weights,lengths_img, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(scores,attn_weights,lengths_img)
        self.logger.update('loss', loss.data, scores.size(0))
        return loss
    def train_emb(self, images, captions, lengths, lengths_img, ids=None, *args):
        """One training step given images and captions.
           输入依次是一个batch的padding后的视频、padding后的文本，文本的单词数目，视频的滑窗数目，pair的序号
        """
        ##print(ids)
        self.Eiters += 1
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        #img_emb, cap_emb, attn_weights = self.forward_emb(images, captions, lengths, lengths_img)
        attn_weights,scores = self.forward_emb(images, captions, lengths, lengths_img)
        # measure accuracy and record loss
        self.optimizer.zero_grad()
        #loss = self.forward_loss(img_emb, cap_emb)
        loss = self.forward_loss(scores,attn_weights,lengths_img)
        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            # 正则化
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

		
		
