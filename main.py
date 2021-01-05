import os
import logging
import argparse
from models.runner import Runner
from tools.vocab import Vocabulary
from easydict import EasyDict as edict
import yaml
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
    parser.add_argument('--learning_rate', default=.01, type=float) # 论文中设为0.001
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--lr_update', default=20, type=int) # 论文中为15
    parser.add_argument('--global_margin', default=0.2, type=float,
                        help='Rank loss margin.') # 0.1 for Charades-STA and 0.2 for DiDeMo
    parser.add_argument('--local_margin', default=0.2, type=float,
                        help='Rank loss margin.') # 0.1 for Charades-STA and 0.2 for DiDeMo
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Size of a training mini-batch.') # 论文中为128
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--joint_dim', default=512, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--video_dim', default=4096, type=int,
                        help='Dimensionality of the video embedding.')
    parser.add_argument('--temporal_scale', default=31, type=int,
                        help='ActivityNet=100，Charades=20') # 视频时序长度
    parser.add_argument('--post_process_thread', default=4, type=int)
    parser.add_argument('--soft_nms_alpha', type=float, default=0.4)
    parser.add_argument('--soft_nms_low_thres',type=float,default=0.5)
    parser.add_argument('--soft_nms_high_thres',type=float,default=0.9)
    parser.add_argument('--raw_feature_norm', default="no_norm",
                        help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
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
    parser.add_argument('--layers', default=4, type=int)
    parser.add_argument('--kernel_size', default=3, type=int)
    parser.add_argument('--stride', default=2, type=int)
    parser.add_argument('--neg_num', default=3, type=int)
    parser.add_argument('--smooth_lam', default=1, type=float)
    parser.add_argument('--start_layer', default=1, type=float)
    parser.add_argument('--start_local', default=10, type=int)
    args = parser.parse_args()
    return args



def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    # 必须为edict对象
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # a中的参数必须是b中有的
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        # 对应参数的类型必须相同
        old_type = type(getattr(b,k))
        if old_type is not type(v):
            if isinstance(getattr(b,k), np.ndarray):
                v = np.array(v, dtype=getattr(b,k).dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) ' 
                                'for config key: {}').format(type(getattr(b,k),type(v), k)))

        # recursively merge dicts
        if type(v) is edict:
            try:
                # 如果参数同样是edict字典，则递归调用
                _merge_a_into_b(a[k], getattr(b,k))
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            # 覆盖b中的值
            setattr(b,k,v)

def cfg_from_file(dataset,opt):
    """Load a config file and merge it into the default options."""
    # 从yaml文件中读取参数，并覆盖默认参数
    if dataset == 'Charades':
        filename = 'data/Charades.yml'
    elif dataset == 'TACoS':
        filename = 'data/TACoS.yml'
    elif dataset == 'ActivityNet':
        filename = 'data/ActivityNet.yml'

    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, opt)
    

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    opt = parse_args()
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    cfg_from_file(opt.dataset,opt)

    # lam_list = [1,0.1,10,0.01,100,0.001,0.0001]
    # for i in lam_list:
    #     opt.smooth_lam = i
    #     opt.model_name = 'tacos_sparse_relu_' +str(i)
    print(opt)
    train_runner = Runner(opt,is_training = True)
    train_runner.train()
    test_runner = Runner(opt, is_training = False)
    test_runner.test(os.path.join(opt.model_path,opt.model_name))