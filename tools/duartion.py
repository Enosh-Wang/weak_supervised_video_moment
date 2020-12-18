import pandas as pd 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import norm,poisson,lognorm

#df = pd.read_csv('/home/share/wangyunxiao/Charades/caption/charades_val.csv')
#df = pd.read_csv('/home/share/wangyunxiao/Charades/caption/charades_train.csv')
#df = pd.read_csv('/home/share/wangyunxiao/Charades/caption/charades_test.csv')

#df = pd.read_csv('/home/share/wangyunxiao/ActivityNet/caption/activitynet_val.csv')
#df = pd.read_csv('/home/share/wangyunxiao/ActivityNet/caption/activitynet_train.csv')
#df = pd.read_csv('/home/share/wangyunxiao/ActivityNet/caption/activitynet_test.csv')

df = pd.read_csv('/home/share/wangyunxiao/TACoS/caption/tacos_train.csv')
duration = df['duration']
start = df['start_time']
end = df['end_time']

d = (end - start)/duration

f = plt.figure(figsize=(6,4))
d = np.asarray(d)
d = np.clip(d,a_min=1e-8,a_max=np.max(d))

# d = np.log(d)
# n,bins,patches = plt.hist(x = d, bins='auto', density = True, rwidth=0.9, color='#607c8e')

# mean = np.mean(d)
# std = np.std(d)

# x = bins+1e-8
# y = np.exp( -(x-mean)**2/(2* (std**2))) / std*np.sqrt(2*np.pi)
# plt.plot(x, y/6, c='m')

# y1 = norm.pdf(bins,mean,std)
# plt.plot(x,y1, c='b')


n,bins,patches = plt.hist(x = d, bins='auto', density = True, rwidth=0.9, color='#607c8e')

d = np.log(d+1e-8)
mean = np.mean(d)
std = np.std(d)


# x = bins+1e-8
# y1 =  np.exp( -( (np.log(x)-mean)**2/(std**2) )/2 ) / std*x*np.sqrt(2*np.pi)
# plt.plot(x, y1, c='m')



y = lognorm.pdf(bins, scale=np.exp(mean), s=std,loc=0)
plt.plot(bins, y, c='b')

plt.title('activitynet_train')
plt.grid(axis='y', alpha=0.75)
plt.savefig('activitynet_train.png')
plt.close(f)


# mean = np.mean(d)
# var = np.var(d)

# mu = np.log(mean) - 0.5*np.log(1+var/(mean**2))
# sig = np.sqrt(np.log(1+var/(mean**2)))
# mean = np.exp(mu)
# std = sig