import numpy as np
import pickle

path = 'output/tacos_sparse_0.1_top10.pkl'
with open(path,'rb') as f:
    res = pickle.load(f)

print(res)