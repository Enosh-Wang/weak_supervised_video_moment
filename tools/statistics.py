import pickle
import numpy as np
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt


activity_path = '/home/share/wangyunxiao/ActivityNet/activitynet_127.pkl'
charades_path = '/home/share/wangyunxiao/Charades/charades_31.pkl'
tacos_path = '/home/share/wangyunxiao/TACoS/tacos_255.pkl'

with open(activity_path,'rb') as f:
    activity_f = pickle.load(f)
with open(charades_path,'rb') as f:
    charades_f = pickle.load(f)
with open(tacos_path,'rb') as f:
    tacos_f = pickle.load(f)

data_a = np.asarray(list(activity_f.values()))
data_c = np.asarray(list(charades_f.values()))
data_t = np.asarray(list(tacos_f.values()))

# print('mean:')
# print(np.mean(data_a))
# print(np.mean(data_c))
# print(np.mean(data_t))

# print('min:')
# print(np.min(data_a))
# print(np.min(data_c))
# print(np.min(data_t))

# print('max:')
# print(np.max(data_a))
# print(np.max(data_c))
# print(np.max(data_t))

# print('var:')
# print(np.var(data_a))
# print(np.var(data_c))
# print(np.var(data_t))

print('std:')
print(np.std(data_a))
print(np.std(data_c))
print(np.std(data_t))

f = plt.figure(figsize=(6,4))
plt.boxplot([data_a.flatten(),data_c.flatten(),data_t.flatten()])
plt.savefig('statistics.png')
plt.close(f)