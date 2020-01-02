import pickle
import os
import time
import shutil

import torch
import pandas
import data_charades as data
from vocab import Vocabulary
from model_charades import VSE
from evaluation_charades import t2i, AverageMeter, LogCollector, encode_data, evalrank

import logging
from torch.utils.tensorboard import SummaryWriter

import argparse
import os

# 从vsepp拷贝的，改写成charades
def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/share/wangyunxiao/Charades',
                        help='path to datasets')
    parser.add_argument('--data_name', default='charades_precomp',
                        help='charades_precomp')
    parser.add_argument('--name', default='default',
                        help='model name')
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--margin', default=0.1, type=float,
                        help='Rank loss margin.') # 0.1 for Charades-STA and 0.2 for DiDeMo
    parser.add_argument('--num_epochs', default=80, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.') # 论文中为128
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.001, type=float,
                        help='Initial learning rate.') # 论文中设为0.001
    parser.add_argument('--lr_update', default=40, type=int,
                        help='Number of epochs to update the learning rate.') # 论文中为15
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/runX',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=4096, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--use_restval', action='store_true',
                        help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--reset_train', action='store_true',
                        help='Ensure the training is always done in '
                        'train mode (Not recommended).')
    opt = parser.parse_args()
    print(opt)
    
    # python 内置日志模块
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    # tensorboard 日志
    tb_writer = SummaryWriter(opt.logger_name, flush_secs=5)

    # Load Vocabulary Wrapper
    # 加载字典
    vocab = pickle.load(open(os.path.join(opt.vocab_path, 'vocab.pkl'), 'rb'))
    opt.vocab_size = len(vocab)

    # Load data loaders
    # 加载数据集
    train_loader, val_loader = data.get_loaders(
        opt.data_name, vocab, opt.crop_size, opt.batch_size, opt.workers, opt)

    # Construct the model
    # 加载模型
    model = VSE(opt)
    #tb_writer.add_graph(model,train_loader)
    start_epoch = 0
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            # 从checkpoint继续训练
            # 加载checkpoint
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            # 恢复参数
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            # 使日志连续
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model,tb_writer)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    best_rsum = 0
    for epoch in range(start_epoch,opt.num_epochs):
        # 重载学习速率，每 30 epoch 除以10，根据重载的epoch数计算当前的学习速率
        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader,tb_writer)

        # evaluate on validation set
        rsum = validate(opt, val_loader, model,tb_writer)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, prefix=opt.logger_name + '/')


def train(opt, train_loader, model, epoch, val_loader ,tb_writer):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        if opt.reset_train:
            # Always reset to train mode, this is not the default behavior
            model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_writer.add_scalar('epoch', epoch, global_step=model.Eiters)
        # 相当于创建了一个字典，先暂存，再一次性保存到writer
        model.logger.tb_log(tb_writer, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            validate(opt, val_loader, model, tb_writer)


def validate(opt, val_loader, model, tb_writer):
    # compute the encoding for all the validation images and captions
    path = os.path.join(opt.data_path, opt.data_name)+"/Caption/charades_val.csv"
    df = pandas.read_csv(open(path,'rb'))

    #img_embs, cap_embs, attn_index, lengths_img = encode_data(model, val_loader)
    attn_index, lengths_img = encode_data(model, val_loader,tb_writer,df,is_training=True)
    # image retrieval
    r13, r15, r17 = t2i(df, attn_index, lengths_img)
    logging.info("Text to image: %.1f, %.1f, %.1f" %
                 (r13, r15, r17))
    # sum of recalls to be used for early stopping
    currscore = r13 + r15 + r17

    # record metrics in tensorboard
    tb_writer.add_scalar('val/r13', r13, global_step=model.Eiters)
    tb_writer.add_scalar('val/r15', r15, global_step=model.Eiters)
    tb_writer.add_scalar('val/r17', r17, global_step=model.Eiters)
    tb_writer.add_scalar('val/rsum', currscore, global_step=model.Eiters)

    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    """保存checkpoint，如果是best，再拷贝一份命名为model_best.pth.tar"""
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
    path = os.path.join('test_charades','cost_p_max+cost_n_max')
    if not os.path.exists(path):
        os.mkdir(path)
    os.system("mv runs/runX/* "+path)
    DATA_PATH = '/home/share/wangyunxiao/Charades'
    RUN_PATH = '/home/wangyunxiao/weak_supervised_video_moment/'
    evalrank(os.path.join(RUN_PATH,path,'model_best.pth.tar'), data_path=DATA_PATH, split="test")

