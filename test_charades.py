from vocab import Vocabulary
import evaluation_charades as evaluation
# 建一个data软连指向数据集
DATA_PATH = '/home/share/wangyunxiao/Charades'

evaluation.evalrank("test_charades/biGRU/model_best.pth.tar", data_path=DATA_PATH, split="test")
