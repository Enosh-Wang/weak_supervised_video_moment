import torch
import numpy as np
from collections import OrderedDict
from torch.backends import cudnn
from models.model import Model
from torch.utils.tensorboard import SummaryWriter
from data.data_loder import get_data_loader
import logging
import pandas
import os
import time
import shutil
from utils.util import AverageMeter,t2i,load_glove
from utils.plot import plot_similarity,plot_pca,plot_sentence,plot_video
import pickle
from utils.vocab import Vocabulary


class Runner(object):

    def __init__(self, opt, is_training):

        self.opt = opt
        self.is_training = is_training
        self.vocab = self.get_vocab()
        self.word2vec = load_glove(opt.glove_path)
        self.opt.vocab_size = len(self.vocab)
        self.grad_clip = self.opt.grad_clip
        self.model = Model(self.opt,self.vocab)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            cudnn.benchmark = True

        self.optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.opt.learning_rate)
        self.logger = SummaryWriter(os.path.join(self.opt.model_path,self.opt.model_name), flush_secs=5)
        self.iters = 0
        self.start_epoch = 0
        self.best_rsum = 0
        self.dataframe = self.get_df()

    def train(self):

        # Load data loaders
        # 加载数据集
        #train_loader = get_data_loader(self.opt, self.vocab, 'train', True)
        #val_loader = get_data_loader(self.opt, self.vocab, 'val', False)
        train_loader = get_data_loader(self.opt, self.word2vec, 'train', True)
        val_loader = get_data_loader(self.opt, self.word2vec, 'val', False)

        # optionally resume from a checkpoint
        if self.opt.resume:
            self.load_model()

        # Train the Model
        for epoch in range(self.start_epoch,self.opt.num_epochs):
            # 重载学习速率，每 30 epoch 除以10，根据重载的epoch数计算当前的学习速率
            self.adjust_learning_rate(epoch)
            # train for one epoch
            self.train_one_epoch(train_loader, epoch)

            # evaluate on validation set
            rsum = self.validate(val_loader)

            # remember best R@ sum and save checkpoint
            is_best = rsum > self.best_rsum
            self.best_rsum = max(rsum, self.best_rsum)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'model': self.model.state_dict(),
                'best_rsum': self.best_rsum,
                'opt': self.opt,
                'iters': self.iters,
            }, is_best, prefix=os.path.join(self.opt.model_path,self.opt.model_name))
    
    def validate(self, val_loader):
        # compute the encoding for all the validation images and captions
        score_index, video_lengths = self.encode_data(val_loader, is_training=True)
        # video retrieval
        r13, r15, r17 = t2i(self.dataframe, score_index, video_lengths, is_training=True)
        logging.info("Text to video: %.1f, %.1f, %.1f" % (r13, r15, r17))
        # sum of recalls to be used for early stopping
        currscore = r13 + r15 + r17
        # record metrics in tensorboard
        self.logger.add_scalar('val/r13', r13, global_step=self.iters)
        self.logger.add_scalar('val/r15', r15, global_step=self.iters)
        self.logger.add_scalar('val/r17', r17, global_step=self.iters)
        self.logger.add_scalar('val/rsum', currscore, global_step=self.iters)

        return currscore
    
    def train_one_epoch(self, train_loader, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (videos, sentences, sentence_lengths, video_lengths, index) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            self.iters += 1

            # Set mini-batch dataset
            if torch.cuda.is_available():
                videos = videos.cuda()
                sentences = sentences.cuda()

            _,_,_,_,loss = self.model.forward(videos, sentences, sentence_lengths, video_lengths, self.opt.margin, is_training=True)
 
            self.optimizer.zero_grad()
            # compute gradient and do SGD step
            loss.backward()
            if self.grad_clip > 0:
                # 正则化
                torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Print log info
            if self.iters % self.opt.log_step == 0:
                logging.info(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time))

            # Record logs in tensorboard
            self.logger.add_scalar('epoch', epoch, global_step=self.iters)
            self.logger.add_scalar('loss', loss, global_step=self.iters)
            self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'])


    def get_df(self):

        if self.opt.dataset == 'Charades':
            if self.is_training == True:
                path = os.path.join(self.opt.data_path, self.opt.dataset)+"/caption/charades_val.csv"
            else:
                path = os.path.join(self.opt.data_path, self.opt.dataset)+"/caption/charades_test.csv"
        elif self.opt.dataset== 'ActivityNet':
            if self.is_training == True:
                path = os.path.join(self.opt.data_path, self.opt.dataset)+"/caption/activitynet_val.csv"
            else:
                path = os.path.join(self.opt.data_path, self.opt.dataset)+"/caption/activitynet_test.csv"
        df = pandas.read_csv(open(path,'rb'))
        return df

    def adjust_learning_rate(self,epoch):
        """Sets the learning rate to the initial LR
        decayed by 10 every 30 epochs"""
        lr = self.opt.learning_rate * (0.1 ** (epoch // self.opt.lr_update))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def load_model(self):
        if os.path.isfile(self.opt.resume):
            print("=> loading checkpoint '{}'".format(self.opt.resume))
            checkpoint = torch.load(self.opt.resume)
            # 恢复参数
            self.start_epoch = checkpoint['epoch']
            self.best_rsum = checkpoint['best_rsum']
            self.model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # 使日志连续
            self.iters = checkpoint['iters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                .format(self.opt.resume, self.start_epoch, self.best_rsum))
        else:
            print("=> no checkpoint found at '{}'".format(self.opt.resume))
    
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar', prefix=''):
        """保存checkpoint，如果是best，再拷贝一份命名为model_best.pth.tar"""
        path = os.path.join(prefix,filename)
        torch.save(state, path)
        if is_best:
            shutil.copyfile(path, os.path.join(prefix,'model_best.pth.tar'))
    
    def get_vocab(self):
        if self.opt.dataset == 'Charades':
            vocab = pickle.load(open(os.path.join(self.opt.vocab_path, 'Charades_vocab.pkl'), 'rb'))
            self.opt.vocab_size = len(vocab)
        elif self.opt.dataset == 'ActivityNet':
            vocab = pickle.load(open(os.path.join(self.opt.vocab_path, 'ActivityNet_vocab.pkl'), 'rb'))
            self.opt.vocab_size = len(vocab)
        return vocab

    def encode_data(self, data_loader, is_training=True):
        """Encode all videos and captions loadable by `data_loader`
        """
        batch_time = AverageMeter()
        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        # 数组，用于保存所有的数据
        data_length = len(data_loader.dataset)
        top10_all = np.zeros((data_length, 10))
        video_lengths_all = np.zeros((data_length))
        video_embeddings_all = np.zeros((data_length, self.opt.joint_dim))
        sentence_embeddings_all = np.zeros((data_length, self.opt.joint_dim))
        similarity_all = [] # 尺寸无法确定
        
        for iters, (videos, sentences, sentence_lengths, video_lengths, index) in enumerate(data_loader):
            if torch.cuda.is_available():
                videos = videos.cuda()
                sentences = sentences.cuda()
            # compute the embeddings
            with torch.no_grad():
                video_embedding,sentence_embedding,caption_predict,similarity,loss = self.model.forward(videos, sentences, sentence_lengths, video_lengths, self.opt.margin, is_training=True)
            # 提取对角线元素
            size = similarity.size()
            similarity_diag = torch.zeros(size[0],size[2])
            for i in range(size[0]):
                similarity_diag[i,:] = similarity[i,i,:]

            # padding 滑窗个数不足十个padding到十个
            if(similarity_diag.size(1)<10):
                similarity_positive=torch.zeros(similarity_diag.size(0),10)
                similarity_positive[:,:similarity_diag.size(1)]=similarity_diag
            else:
                similarity_positive=similarity_diag

            batch_length=similarity_positive.size(0)
            
            # 保留每个batch中每个样本的前十个结果的序号
            video_embedding = video_embedding.data.cpu().numpy().copy()
            similarity_positive = similarity_positive.data.cpu().numpy().copy()

            video_embedding_temp = np.zeros((batch_length,self.opt.joint_dim))
            top10= np.zeros((batch_length, 10), dtype=int)

            for k in range(batch_length):
                temp=similarity_positive[k,:]
                top10[k,:]=np.argsort(-temp)[0:10]
                video_embedding_temp[k,:]=video_embedding[k,top10[k,0],:]
        
            # preserve the embeddings by copying from gpu and converting to numpy
            top10_all[index] = top10
            video_lengths_all[index] = video_lengths
            video_embeddings_all[index] = video_embedding_temp
            sentence_embeddings_all[index] = sentence_embedding.data.cpu().numpy().copy()
            similarity_all.append((index,similarity_positive))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if iters % self.opt.log_step == 0:
                logging.info('Test: [{0}/{1}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             .format(iters, len(data_loader), batch_time=batch_time))
            del videos, sentences
        '''
        if is_training == False:
            plot_similarity(similarity_all,video_lengths_all,top10_all[:,0],self.dataframe)
            plot_pca(video_embeddings_all,sentence_embeddings_all,self.dataframe)
            plot_sentence(sentence_embeddings_all,self.dataframe)
            plot_video(video_embeddings_all,video_lengths_all,self.dataframe)
        '''
        return  top10_all, video_lengths_all

    def test(self,model_path):
        # optionally resume from a checkpoint
        checkpoint = torch.load(os.path.join(model_path,'model_best.pth.tar'))
        opt = checkpoint['opt']
        print(opt)
        print('Loading dataset')
        test_loader = get_data_loader(self.opt, self.word2vec, 'test', False)
        self.model.load_state_dict(checkpoint['model'])
        print('Computing results...')
        score_index, video_lengths = self.encode_data(test_loader, is_training=False)
        _, _, _ = t2i(self.dataframe, score_index, video_lengths, is_training=False)


