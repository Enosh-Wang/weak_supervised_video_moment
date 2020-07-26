import os
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
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--glove_path', default='/home/share/wangyunxiao/Glove/glove.840B.300d/glove.840B.300d.pkl',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--learning_rate', default=.002, type=float) # 论文中设为0.001
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--lr_update', default=20, type=int) # 论文中为15
    parser.add_argument('--global_margin', default=0.2, type=float,
                        help='Rank loss margin.') # 0.1 for Charades-STA and 0.2 for DiDeMo
    parser.add_argument('--local_margin', default=0.2, type=float,
                        help='Rank loss margin.') # 0.1 for Charades-STA and 0.2 for DiDeMo
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Size of a training mini-batch.') # 论文中为128
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--joint_dim', default=512, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--video_dim', default=4096, type=int,
                        help='Dimensionality of the video embedding.')
    parser.add_argument('--temporal_scale', default=20, type=int,
                        help='ActivityNet=100，Charades=20') # 视频时序长度
    parser.add_argument('--prop_boundary_ratio', default=0.5, type=float) # proposal拓展率
    parser.add_argument('--start_ratio', default=0, type=float) # proposal拓展率
    parser.add_argument('--end_ratio', default=0.5, type=float) # proposal拓展率
    parser.add_argument('--num_sample', default=6, type=int,
                        help='ActivityNet=32，Charades=6') # 采样点数目
    parser.add_argument('--num_sample_perbin', default=3, type=int) #  子采样点数目
    parser.add_argument('--post_process_thread', default=4, type=int)
    parser.add_argument('--soft_nms_alpha', type=float, default=0.4)
    parser.add_argument('--soft_nms_low_thres',type=float,default=0.5)
    parser.add_argument('--soft_nms_high_thres',type=float,default=0.9)
    parser.add_argument('--raw_feature_norm', default="no_norm",
                        help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
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
    parser.add_argument('--grad_clip', default=5., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--dropout', default=0, type=float,
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
    parser.add_argument('--lambda_lse', default=6., type=float,
                        help='LogSumExp temp.')
    parser.add_argument('--lambda_softmax', default=3., type=float,
                        help='Attention softmax temperature.')
    parser.add_argument('--agg_func', default="Max",
                        help='LogSumExp|Mean|Max|Sum')
    parser.add_argument('--continuation_func', default="Log",
                        help='Linear|Plinear|Sigmoid|Log|Exp')
    parser.add_argument('--model_mode', default="image_IMRAM", type=str,
                        help='full_IMRAM|text_IMRAM')
    parser.add_argument('--iteration_step', default=3, type=int,
                        help='routing_step')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    opt = parse_args()
    print(opt)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    train_runner = Runner(opt,is_training = True)
    train_runner.train()
    test_runner = Runner(opt, is_training = False)
    test_runner.test(os.path.join(opt.model_path,opt.model_name))

