import torch
import numpy as np
from torch.backends import cudnn
from models.model import Model
from torch.utils.tensorboard import SummaryWriter
from data.data_loder import get_data_loader
import logging
import pandas
import os
import time
import shutil
from tools.util import AverageMeter,t2i,load_glove
from tools.plot import plot_similarity,plot_pca,plot_sentence,plot_video,plot_map
from tools.vocab import Vocabulary
from tools.post_processing import post_processing
import pickle

from apex import amp

class Runner(object):

    def __init__(self, opt, is_training):

        self.opt = opt
        self.is_training = is_training
        self.word2vec = load_glove(opt.glove_path)
        self.vocab = self.get_vocab()
        self.grad_clip = opt.grad_clip
        self.model = Model(opt,self.vocab)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            cudnn.benchmark = True

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=opt.learning_rate,weight_decay=opt.weight_decay,momentum=0.9)
        self.model, self.optimizer = amp.initialize(self.model, optimizer, opt_level="O2")
        self.logger = SummaryWriter(os.path.join(opt.model_path,opt.model_name), flush_secs=5)
        self.iters = 0
        self.start_epoch = 0
        self.max_recall = 0
        self.dataframe = self.get_df()

    def train(self):

        # Load data loaders
        # 加载数据集
        train_loader = get_data_loader(self.opt, self.word2vec, self.vocab, 'train', True)
        val_loader = get_data_loader(self.opt, self.word2vec, self.vocab, 'val', True)
        max_iter = len(train_loader)*self.opt.num_epochs+100
        # optionally resume from a checkpoint
        if self.opt.resume:
            self.load_model()

        # Train the Model
        for epoch in range(self.start_epoch,self.opt.num_epochs):
            # 重载学习速率，每 30 epoch 除以10，根据重载的epoch数计算当前的学习速率
            self.adjust_learning_rate(epoch)
            # train for one epoch
            self.train_one_epoch(train_loader, epoch, max_iter)

            # evaluate on validation set
            recall = self.validate(val_loader, max_iter)

            # remember best R@ sum and save checkpoint
            is_best = recall > self.max_recall

            self.max_recall = max(recall, self.max_recall)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'model': self.model.state_dict(),
                'max_recall': self.max_recall,
                'opt': self.opt,
                'iters': self.iters,
            }, is_best, prefix=os.path.join(self.opt.model_path,self.opt.model_name))
    
    def validate(self, val_loader, max_iter):

        # switch to evaluate mode
        self.model.eval()
        val_loss = 0

        batch_time = AverageMeter()
        end = time.time()
        all_result = {}
        for iters, (videos, sentences, sentence_lengths, index, video_name, word_id) in enumerate(val_loader):
            if torch.cuda.is_available():
                videos = videos.cuda().half()
                sentences = sentences.cuda().half()
            # compute the embeddings
            with torch.no_grad():
                confidence_map,loss = self.model.forward(videos, sentences, sentence_lengths, video_name, word_id, self.logger, self.iters, max_iter)
            
            confidence_map = confidence_map.detach().cpu().numpy()
            #plot_map(confidence_map,index,self.opt.model_name)
            batch_result = self.generate_proposal(confidence_map, index)
            
            all_result.update(batch_result)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            val_loss += loss

            if iters % self.opt.log_step == 0:
                logging.info('Test: [{0}/{1}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             .format(iters, len(val_loader), batch_time=batch_time))
            del videos, sentences

        val_loss = val_loss/len(val_loader)
        self.logger.add_scalar('val/loss', val_loss, global_step=self.iters)

        # 保存路径
        path = os.path.join("./output/")
        if not os.path.exists(path):
            os.makedirs(path)
        filename = os.path.join(path,self.opt.model_name+'.pkl')
        with open(filename,'wb') as f:
            pickle.dump(all_result,f)

        top10_filename = post_processing(filename,self.opt)
        R1IOU03, R1IOU05, R1IOU07 = t2i(self.dataframe, top10_filename, is_training=False)
        self.logger.add_scalar('val/R1IOU03', R1IOU03, global_step=self.iters)
        self.logger.add_scalar('val/R1IOU05', R1IOU05, global_step=self.iters)
        self.logger.add_scalar('val/R1IOU07', R1IOU07, global_step=self.iters)
        self.logger.add_scalar('val/R1SUM', R1IOU03+R1IOU05+R1IOU07, global_step=self.iters)

        return R1IOU03+R1IOU05+R1IOU07
    
    def train_one_epoch(self, train_loader, epoch, max_iter):
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (videos, sentences, sentence_lengths, index, video_name, word_id) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            self.iters += 1

            # Set mini-batch dataset
            if torch.cuda.is_available():
                videos = videos.cuda()
                sentences = sentences.cuda()

            _,loss = self.model.forward(videos, sentences, sentence_lengths, video_name, word_id, self.logger, self.iters, max_iter)
 
            self.optimizer.zero_grad()
            # compute gradient and do SGD step

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            #loss.backward()

            if self.grad_clip > 0:
                # 正则化
                torch.nn.utils.clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), self.grad_clip)
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
                self.logger.add_histogram('videos', videos, global_step=self.iters)
                self.logger.add_histogram('sentences', sentences, global_step=self.iters)

            # Record logs in tensorboard
            self.logger.add_scalar('epoch', epoch, global_step=self.iters)
            self.logger.add_scalar('loss', loss, global_step=self.iters)
            self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'], global_step=self.iters)
            


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
            self.max_recall = checkpoint['max_recall']
            self.model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # 使日志连续
            self.iters = checkpoint['iters']
            print("=> loaded checkpoint '{}' (epoch {}, max_recall {})"
                .format(self.opt.resume, self.start_epoch, self.max_recall))
        else:
            print("=> no checkpoint found at '{}'".format(self.opt.resume))
    
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar', prefix=''):
        """保存checkpoint，如果是best，再拷贝一份命名为model_best.pth.tar"""
        path = os.path.join(prefix,filename)
        torch.save(state, path)
        if is_best:
            shutil.copyfile(path, os.path.join(prefix,'model_best.pth.tar'))

    def test(self, model_path, max_iter=5700):
        # optionally resume from a checkpoint
        checkpoint = torch.load(os.path.join(model_path,'checkpoint.pth.tar'))
        opt = checkpoint['opt']
        print(opt)
        print('Loading dataset')
        test_loader = get_data_loader(self.opt, self.word2vec, self.vocab, 'test', False)
        self.model.load_state_dict(checkpoint['model'])
        print('Computing results...')

        # switch to evaluate mode
        self.model.eval()
        batch_time = AverageMeter()
        end = time.time()
        all_result = {}
        for iters, (videos, sentences, sentence_lengths, index, video_name, word_id) in enumerate(test_loader):
            if torch.cuda.is_available():
                videos = videos.cuda().half()
                sentences = sentences.cuda().half()
            # compute the embeddings
            with torch.no_grad():
                self.iters += 1
                confidence_map,loss = self.model.forward(videos, sentences, sentence_lengths, video_name, word_id, self.logger, self.iters, max_iter)

            confidence_map = confidence_map.detach().cpu().numpy()
            plot_map(confidence_map,index,self.opt.model_name)
            batch_result = self.generate_proposal(confidence_map, index)
            
            all_result.update(batch_result)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if iters % self.opt.log_step == 0:
                logging.info('Test: [{0}/{1}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             .format(iters, len(test_loader), batch_time=batch_time))
            del videos, sentences

        # 保存路径
        path = os.path.join("./output/")
        if not os.path.exists(path):
            os.makedirs(path)
        filename = os.path.join(path,self.opt.model_name+'.pkl')
        with open(filename,'wb') as f:
            pickle.dump(all_result,f)

        top10_filename = post_processing(filename,self.opt)
        _, _, _ = t2i(self.dataframe, top10_filename, is_training=False)

    def generate_proposal(self, score_map, index):
        batch_size = score_map.shape[0]
        tscale = self.opt.temporal_scale
        batch_result = {}
        for i in range(batch_size):
            results = []
            for idx in range(tscale):
                for jdx in range(tscale):
                    # 起止点的索引
                    start_index = jdx
                    end_index = start_index + idx+1
                    # 如果是上一步中选定的起止点，则进一步验证置信度
                    if end_index < tscale :
                        # 置信度得分
                        score = score_map[i,idx, jdx]
                        # 保存proposal 坐标是百分比
                        if score > -1000:
                            # proposal的坐标
                            xmin = start_index/tscale
                            xmax = end_index/tscale
                            results.append([xmin, xmax, score])
            results = np.stack(results)
            batch_result[index[i]] = results
        return batch_result

    def l2_Regular(self): 
        l2_reg = torch.tensor(0.).cuda()
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad == True:
                l2_reg += torch.norm(param) 
        return l2_reg * self.opt.weight_decay
    
    def get_vocab(self):
        if self.opt.dataset == 'Charades':
            vocab = pickle.load(open(os.path.join(self.opt.vocab_path, 'Charades_vocab.pkl'), 'rb'))
            self.opt.vocab_size = len(vocab)
        elif self.opt.dataset == 'ActivityNet':
            vocab = pickle.load(open(os.path.join(self.opt.vocab_path, 'ActivityNet_vocab.pkl'), 'rb'))
            self.opt.vocab_size = len(vocab)
        return vocab
    
    
        




