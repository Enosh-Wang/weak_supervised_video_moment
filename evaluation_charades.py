from __future__ import print_function
import os
import pickle

import numpy
from data_charades import get_test_loader
import time
import numpy as np
from vocab import Vocabulary 
import torch
from model_charades import VSE
from collections import OrderedDict
import pandas
from torch.utils.tensorboard import SummaryWriter
from plot_util import plot_pca,plot_similarity,plot_sentence,plot_video

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.add_scalar(prefix + k, v.val, global_step=step)

def cIoU(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
    return float(intersection)/union

def encode_data(model, data_loader,tb_writer,df, is_training=True, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()
    # 数组，用于保存所有epoch的数据
    attention_index_all = np.zeros((len(data_loader.dataset), 10))
    rank1_index_all = np.zeros((len(data_loader.dataset)))
    lengths_all = np.zeros((len(data_loader.dataset)))
    img_embs_all = np.zeros((len(data_loader.dataset), 1024))
    cap_embs_all = np.zeros((len(data_loader.dataset), 1024))
    attention_weight_all = np.zeros((len(data_loader.dataset),100)) # 统计最大21
    

    for epoch, (images, captions, lengths, lengths_img, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        with torch.no_grad(): # 替代volatile=True，已弃用
            img_emb, cap_emb, attenton_weight = model.forward_emb(images, captions, lengths, lengths_img, is_training)
        # 提取对角线元素
        size = attenton_weight.size()
        attenton_weight_diag = torch.zeros(size[0],size[2])
        for i in range(size[0]):
            attenton_weight_diag[i,:] = attenton_weight[i,i,:]

        # padding 滑窗个数不足十个padding到十个 为了方便合成一个数组进行可视化，改成25了
        if(attenton_weight_diag.size(1)<100):
            attenton_weight_positive=torch.zeros(attenton_weight_diag.size(0),100)
            attenton_weight_positive[:,0:attenton_weight_diag.size(1)]=attenton_weight_diag
        else:
            attenton_weight_positive=attenton_weight_diag

        batch_length=attenton_weight_positive.size(0)
        attenton_weight_positive=torch.squeeze(attenton_weight_positive)
        
        # 保留每个batch中每个样本的前十个结果的序号
        attn_index= np.zeros((batch_length, 10)) # Rank 1 to 10
        rank_att1= np.zeros(batch_length)
        img_emb_batch = img_emb.data.cpu().numpy().copy()
        img_emb_temp = np.zeros((batch_length,1024))
        temp=attenton_weight_positive.data.cpu().numpy().copy()
        for k in range(batch_length):
            att_weight=temp[k,:]
            sc_ind=numpy.argsort(-att_weight)
            rank_att1[k]=sc_ind[0]
            attn_index[k,:]=sc_ind[0:10]
            img_emb_temp[k,:]=img_emb_batch[k,sc_ind[0],:]
	
        # preserve the embeddings by copying from gpu and converting to numpy
        attention_index_all[ids] = attn_index
        lengths_all[ids] = lengths_img
        rank1_index_all[ids] = rank_att1
        img_embs_all[ids] = img_emb_temp
        cap_embs_all[ids] = cap_emb.data.cpu().numpy().copy()
        attention_weight_all[ids] = temp

        # measure accuracy and record loss
        model.forward_loss(attenton_weight,lengths_img)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if epoch % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        epoch, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del images, captions
    if is_training == False:
        plot_similarity(attention_weight_all,lengths_all,rank1_index_all,df)
        plot_pca(img_embs_all,cap_embs_all,df)
        plot_sentence(cap_embs_all,df)
        plot_video(img_embs_all,lengths_all,df)
    return  attention_index_all, lengths_all


def evalrank(model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model.
    """
    # load model and options
    checkpoint = torch.load(model_path) # “model_best.pth.tar”
    opt = checkpoint['opt'] 

    tb_writer = SummaryWriter(opt.logger_name, flush_secs=5)

    if data_path is not None:
        opt.data_path = data_path
    opt.vocab_path = "./vocab/"
    # load vocabulary	   
    vocab = pickle.load(open(os.path.join(
        opt.vocab_path, 'vocab.pkl'), 'rb'))

    opt.vocab_size = len(vocab)

    # construct model
    model = VSE(opt)
    
    # load model state
    model.load_state_dict(checkpoint['model'])
    print(opt)	
	
    ####### input video files
    path= os.path.join(opt.data_path, opt.data_name)+"/Caption/charades_"+ str(split) + ".csv"
    df=pandas.read_csv(open(path,'rb'))

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab, opt.crop_size,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    attn_index, lengths_img = encode_data(model, data_loader,tb_writer,df,is_training=False)

	# retrieve moments
    _, _, _ = t2i(df, attn_index, lengths_img)

def t2i( df, attn_index, lengths_img, npts=None):
    """
    Text->Images (Image Search)
    Images: (N, K) matrix of images
    Captions: (N, K) matrix of captions
    """
    # 读取GT
    start_segment=df['start_segment']
    end_segment=df['end_segment']
	
    if npts is None:
        # pair的数目？
        npts = attn_index.shape[0]

    correct_num05=0
    correct_num07=0
    correct_num03=0

    R5IOU5=0
    R5IOU7=0
    R5IOU3=0
    R10IOU3=0
    R10IOU5=0
    R10IOU7=0
	
    for index in range(int(npts)):
        # index应该是指pair的索引
        # 匹配结果的排序列表
        att_inds=attn_index[index,:]
        # 视频的长度
        len_img=lengths_img[index]
        # 读取GT
        gt_start=start_segment[index]
        gt_end=end_segment[index]
        # 因为将128和256的滑窗拼接在了一起，所以前2/3的特征是128滑窗的，后1/3的特征是256滑窗的
        break_128=np.floor(len_img*2/3)
        # 起始帧
        rank1_start=att_inds[0]
        if (rank1_start<break_128):
           # 128的滑窗
           rank1_start_seg =rank1_start*128
           rank1_end_seg = rank1_start_seg+128
        else:
           # 256的滑窗
           rank1_start_seg =(rank1_start-break_128)*256
           rank1_end_seg = rank1_start_seg+256
        
        # 计算IOU
        iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
        if iou>=0.5:
           correct_num05+=1
        if iou>=0.7:
           correct_num07+=1
        if iou>=0.3:
           correct_num03+=1

        for j1 in range(5):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.5:
               R5IOU5+=1
               break
			   
        for j1 in range(5):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.7:
               R5IOU7+=1
               break
			   
        for j1 in range(5):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.3:
               R5IOU3+=1
               break
			   
        for j1 in range(10):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.5:
               R10IOU5+=1
               break
			   
        for j1 in range(10):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.7:
               R10IOU7+=1
               break
			   
        for j1 in range(10):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.3:
               R10IOU3+=1
               break    
			    	
			   
	
	
	############################

    # Compute metrics
    R1IoU05=correct_num05
    R1IoU07=correct_num07
    R1IoU03=correct_num03
    total_length=attn_index.shape[0]
    #print('total length',total_length)
    print("R@1 IoU0.3: %f" %(R1IoU03/float(total_length)))
    print("R@5 IoU0.3: %f" %(R5IOU3/float(total_length)))
    print("R@10 IoU0.3: %f" %(R10IOU3/float(total_length)))
	
    print("R@1 IoU0.5: %f" %(R1IoU05/float(total_length)))
    print("R@5 IoU0.5: %f" %(R5IOU5/float(total_length)))
    print("R@10 IoU0.5: %f" %(R10IOU5/float(total_length)))
	
    print("R@1 IoU0.7: %f" %(R1IoU07/float(total_length)))
    print("R@5 IoU0.7: %f" %(R5IOU7/float(total_length)))
    print("R@10 IoU0.7: %f" %(R10IOU7/float(total_length)))
	
	
    return R1IoU03, R1IoU05, R1IoU07
		
		