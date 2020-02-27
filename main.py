import os
import time
import logging
import argparse
from models.runner import Runner
from tools.vocab import Vocabulary
# 从vsepp拷贝的，改写成charades
def parse_args():
    '''Hyper Parameters'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/share/wangyunxiao',
                        help='path to datasets') # 正式版在根目录下见一个dataset文件夹，里面用软链链到share目录
    parser.add_argument('--dataset', default='Charades',
                        help='Charades,ActivityNet')
    parser.add_argument('--model_path', default='runs',
                    help='Path to save the model and Tensorboard log.')
    parser.add_argument('--model_name', default='default',
                        help='model name')
    parser.add_argument('--num_epochs', default=50, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', default=.001, type=float,
                        help='Initial learning rate.') # 论文中设为0.001
    parser.add_argument('--lr_update', default=25, type=int,
                        help='Number of epochs to update the learning rate.') # 论文中为15
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--glove_path', default='/home/share/wangyunxiao/Glove/glove.840B.300d/glove.840B.300d.pkl',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--global_margin', default=0.1, type=float,
                        help='Rank loss margin.') # 0.1 for Charades-STA and 0.2 for DiDeMo
    parser.add_argument('--local_margin', default=0.1, type=float,
                        help='Rank loss margin.') # 0.1 for Charades-STA and 0.2 for DiDeMo
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Size of a training mini-batch.') # 论文中为128
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--joint_dim', default=512, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--video_dim', default=4096, type=int,
                        help='Dimensionality of the video embedding.')
    parser.add_argument('--temporal_scale', default=20, type=int,
                        help='ActivityNet=100，Charades=20') # 视频时序长度
    parser.add_argument('--prop_boundary_ratio', default=0.5, type=float,
                        help='') # proposal拓展率
    parser.add_argument('--num_sample', default=6, type=int,
                        help='ActivityNet=32，Charades=6') # 采样点数目
    parser.add_argument('--num_sample_perbin', default=3, type=int) #  子采样点数目
    parser.add_argument('--negative_num', default=5, type=int)
    parser.add_argument('--post_process_thread', default=4, type=int)
    parser.add_argument('--soft_nms_alpha', type=float, default=0.4)
    parser.add_argument('--soft_nms_low_thres',type=float,default=0.5)
    parser.add_argument('--soft_nms_high_thres',type=float,default=0.9)
    parser.add_argument('--sentence_heads', default=4, type=int,
                        help='')
    parser.add_argument('--video_heads', default=8, type=int,
                        help='')
    parser.add_argument('--num_sets', default=12, type=int,
                        help='')
    parser.add_argument('--sentence_attn_layers', default=1, type=int,
                        help='')
    parser.add_argument('--video_attn_layers', default=1, type=int,
                        help='')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='The dropout value.')
    parser.add_argument('--RNN_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the video embeddings.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    opt = parse_args()
    print(opt)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    train_runner = Runner(opt,is_training = True)
    train_runner.train()
    test_runner = Runner(opt, is_training = False)
    test_runner.test(os.path.join(opt.model_path,opt.model_name))


