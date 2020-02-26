from stanfordcorenlp import StanfordCoreNLP
import pandas as pd
import numpy as np
import json

#df = pd.read_csv('/home/share/wangyunxiao/Charades/caption/charades_val.csv')
#df = pd.read_csv('/home/share/wangyunxiao/Charades/caption/charades_train.csv')
df = pd.read_csv('/home/share/wangyunxiao/Charades/caption/charades_test.csv')

relation = {'empty': 0, 'self_loop': 1, 'nsubj': 2, 'det': 3, 'aux': 4, 'cc': 5, 'conj': 6, 'nmod': 7, 'case': 8, 'dobj': 9, 'amod': 10, 'compound': 11, 'ccomp': 12, 'advmod': 13, 'mark': 14, 'advcl': 15, 'nummod': 16, 'acl': 17, 'nmod:poss': 18, 'dep': 19, 'nsubjpass': 20, 'auxpass': 21, 'xcomp': 22, 'appos': 23, 'acl:relcl': 24, 'compound:prt': 25, 'cop': 26, 'neg': 27, 'mwe': 28, 'parataxis': 29, 'expl': 30, 'det:predet': 31, 'csubjpass': 32, 'nmod:tmod': 33, 'nmod:npmod': 34, 'iobj': 35, 'root': 36, 'cc:preconj': 37, 'csubj': 38, 'punct': 39, 'discourse': 40}
all_words = []
all_id2pos = []
all_mat = []

nlp = StanfordCoreNLP('/home/share/wangyunxiao/stanford-corenlp-full-2018-10-05')
for sentence in df['description']:
    words = nlp.word_tokenize(sentence)
    id2pos = list(range(len(words)))
    res = nlp.dependency_parse(sentence)
    mat = np.eye(len(words),dtype=int)
    for item in res:
        if item[0] == 'ROOT':
            continue
        mat[item[1]-1,item[2]-1] = relation[item[0]]

    all_words.append(words)
    all_id2pos.append(id2pos)
    all_mat.append(mat)

nlp.close()
df['words'] = all_words
df['id2pos'] = all_id2pos
df['mat'] = all_mat
#df.to_csv('/home/share/wangyunxiao/Charades/caption/charades_val_nlp.csv')
#df.to_csv('/home/share/wangyunxiao/Charades/caption/charades_train_nlp.csv')
df.to_csv('/home/share/wangyunxiao/Charades/caption/charades_test_nlp.csv')