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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

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

    attention_index = np.zeros((len(data_loader.dataset), 10))
    rank1_ind = np.zeros((len(data_loader.dataset)))
    lengths_all = np.zeros((len(data_loader.dataset)))

    pre_128 = 0
    pre_256 = 0
    gt_128 = 0
    gt_256 = 0
    too_small = 0
    too_large = 0
    local_max = 0
    small_std = 0
    too_small_iou = 0

    for epoch, (images, captions, lengths, lengths_img, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        with torch.no_grad(): # 替代volatile=True，已弃用
            attn_weights = model.forward_emb(images, captions, lengths, lengths_img)
		
        # 提取对角线元素
        size = attn_weights.size()
        attn_weights_diag = torch.zeros(size[0],size[2])
        for i in range(size[0]):
            attn_weights_diag[i,:] = attn_weights[i,i,:]
        # padding
        if(attn_weights_diag.size(1)<10):
            attn_weight=torch.zeros(attn_weights_diag.size(0),10)
            attn_weight[:,0:attn_weights_diag.size(1)]=attn_weights_diag
        else:
            attn_weight=attn_weights_diag

        batch_length=attn_weight.size(0)
        attn_weight=torch.squeeze(attn_weight)
        
        # 保留每个batch中每个样本的前十个结果
        attn_index= np.zeros((batch_length, 10)) # Rank 1 to 10
        rank_att1= np.zeros(batch_length)
        temp=attn_weight.data.cpu().numpy().copy()
        for k in range(batch_length):
            att_weight=temp[k,:]
            sc_ind=numpy.argsort(-att_weight)
            rank_att1[k]=sc_ind[0]
            attn_index[k,:]=sc_ind[0:10]
	
        # preserve the embeddings by copying from gpu and converting to numpy
        attention_index[ids] = attn_index
        lengths_all[ids] = lengths_img
        rank1_ind[ids] = rank_att1

        # measure accuracy and record loss
        model.forward_loss(attn_weights,lengths_img)

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
            # 绘图
            path = os.path.join('img',str(epoch))
            if not os.path.exists(path):
                os.makedirs(path)
            
            # 读取GT
            start_segment=df['start_segment']
            end_segment=df['end_segment']
            attn_max = torch.max(attn_weight).data
            attn_weight = attn_weight.data.cpu().numpy().copy()
            for index in range(batch_length):

                # 视频的长度
                len_img=lengths_img[index]

                break_128=int(np.floor(len_img*2/3))

                gt_start = start_segment[ids[index]]
                gt_end = end_segment[ids[index]]

                # 计算GT和所有滑窗的iou
                start_128 = range(break_128)
                end_128 = range(1,int(break_128+1))
                start_256 = range(int(len_img-break_128))
                end_256 = range(1,int(len_img-break_128+1))
                iou_list=[]
                for start,end in zip(start_128,end_128):
                    iou = cIoU((gt_start,gt_end),(start*128,end*128))
                    iou_list.append(iou)
                for start,end in zip(start_256,end_256):
                    iou = cIoU((gt_start,gt_end),(start*256,end*256))
                    iou_list.append(iou)
                iou_array = np.array(iou_list)
                iou_max = np.argmax(iou_array)

                # 统计128窗口的得分分布情况
                # 标准差
                score = attn_weight[index,:len_img]
                score_std = np.std(score)
                if score_std < 0.025 :
                    small_std += 1

                # 局部最大值
                score_argmax = np.argmax(score)
                score_max = np.max(score)
                local_max_cnt = 0
                for i in range(len_img):
                    if i == score_argmax:
                        continue
                    elif i == 0 :
                        if score[0] > score[1] and score[0] > score_max-0.01:
                            local_max_cnt += 1
                    elif i == len_img-1:
                        if score[len_img-1] > score[len_img-2] and score[len_img-1] > score_max-0.01:
                            local_max_cnt += 1
                    else:
                        if score[i] > score[i-1] and score[i] > score[i+1] and score[i] > score_max-0.01:
                            local_max_cnt += 1
                if local_max_cnt > 0:
                    local_max += 1


                # 起始帧
                rank1_start=rank_att1[index]
                if (rank1_start<break_128):
                    # 128的滑窗
                    pre_128 += 1
                    rank1_start_seg =rank1_start*128
                    rank1_end_seg = rank1_start_seg+128
                else:
                    # 256的滑窗
                    pre_256 += 1
                    rank1_start_seg =(rank1_start-break_128)*256
                    rank1_end_seg = rank1_start_seg+256

                gt_start_128 = gt_start/128.0
                gt_end_128 = gt_end/128.0
                gt_start_256 = gt_start/256.0 + break_128
                gt_end_256 = gt_end/256.0 + break_128
                if iou_max < break_128 :
                    gt_128 += 1
                    is_gt_128 = True
                else:
                    gt_256 += 1
                    is_gt_128 = False
                
                iou = cIoU((gt_start,gt_end),(rank1_start_seg,rank1_end_seg))
                if iou_max < break_128:
                    if rank1_start >= break_128:
                        too_large += 1
                else:
                    if rank1_start < break_128:
                        too_small += 1
                        if iou > 0:
                            too_small_iou +=1

                f = plt.figure(figsize=(6,4))
                # 绘制分界线
                plt.plot([break_128,break_128],[0,attn_max],linestyle=":",color='gray')
                plt.plot([len_img,len_img],[0,attn_max],linestyle=":",color='gray')
                # 绘制预测区域
                plt.plot([rank1_start,rank1_start+1],[attn_max*0.6,attn_max*0.6],linewidth=4,color='darkred')
                # 绘制GT区域
                if is_gt_128:
                    plt.plot([gt_start_128,gt_end_128],[attn_max*0.4,attn_max*0.4],linewidth=4,color='orange',marker='o')
                    plt.plot([gt_start_256,gt_end_256],[attn_max*0.4,attn_max*0.4],linewidth=4,color='orange')
                else:
                    plt.plot([gt_start_128,gt_end_128],[attn_max*0.4,attn_max*0.4],linewidth=4,color='orange')
                    plt.plot([gt_start_256,gt_end_256],[attn_max*0.4,attn_max*0.4],linewidth=4,color='orange',marker='o')
                # 绘制得分曲线
                '''
                x_128 = range(break_128)
                y_128 = score[x_128]
                plt.plot(x_128,y_128,'g--',marker='o')

                x_256 = range(break_128,len_img)
                y_256 = score[x_256]
                plt.plot(x_256,y_256,'g--',marker='o')
                '''
                x = range(len_img)
                y = score[x]
                plt.plot(x,y,'g--',marker='o')

                
                plt.xlabel('len: %d,  break_128: %d,  rank1: %d,  std: %0.3f,  local_max: %d'
                            %(len_img,break_128,rank1_start,score_std,local_max_cnt))
                plt.title('GT: %d-%d,  Pre: %d-%d,  IOU: %0.2f'%(gt_start,gt_end,rank1_start_seg,rank1_end_seg,iou))
                plt.savefig(os.path.join(path,str(ids[index])+'.png'))
                plt.close(f)
    if is_training == False:
        print('pre_128: %d,  pre_256: %d,  gt_128: %d,  gt_256: %d,  too_small: %d,  too_large: %d,  small_std: %d,  local_max: %d,  too_small_iou: %d'
            %(pre_128,pre_256,gt_128,gt_256,too_small,too_large,small_std,local_max,too_small_iou))
    return  attention_index, lengths_all


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
        break_128=np.floor(len_img*2/3)-1
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
		
		