from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pickle

glove_path = '/home/share/wangyunxiao/Glove/glove.840B.300d/glove.840B.300d.txt'
glove_file = datapath(glove_path)
tmp_file = get_tmpfile("test_word2vec.txt")
_ = glove2word2vec(glove_file, tmp_file)
word2vec = KeyedVectors.load_word2vec_format(tmp_file)
with open('/home/share/wangyunxiao/Glove/glove.840B.300d/glove.840B.300d.pkl', 'wb') as f:
    pickle.dump(word2vec, f)
