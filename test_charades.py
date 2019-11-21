from vocab import Vocabulary
import evaluation_charades as evaluation
# 建一个data软连指向数据集
DATA_PATH = '/home/share/wangyunxiao/Charades'
RUN_PATH = '/home/wangyunxiao/weak_supervised_video_moment/'

evaluation.evalrank(RUN_PATH+"test_charades/video-trans2/checkpoint.pth.tar", data_path=DATA_PATH, split="test")
