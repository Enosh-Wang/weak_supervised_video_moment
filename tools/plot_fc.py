from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import os

model_name = 'multi_cat'
path = os.path.join('runs/',model_name,'model_best.pth.tar')
checkpoint = torch.load(path)['model']

weight_list = []
bias_list = []

for name,param in checkpoint.items():
    if 'loss' in name:
        if 'weight' in name:
            weight_list.append(param.view(-1))
        else:
            bias_list.append(param)

weight = torch.stack(weight_list).cpu().numpy()
bias = torch.stack(bias_list).cpu().numpy()

weight_pca = PCA(n_components=2).fit_transform(weight)
bias_pca = PCA(n_components=2).fit_transform(bias)

marker_list = ['.','<','1','8','s','p','*','h','+','x','d','_']

f = plt.figure(figsize=(6,4))
plt.title('weight')
save_name = model_name + 'weight.png'

for i in range(4):
    marker = marker_list[i]
    plt.scatter(weight_pca[i][0],weight_pca[i][1],c='g',alpha=0.5,s=40,marker=marker)
    plt.scatter(weight_pca[i+4][0],weight_pca[i+4][1],c='c',alpha=0.5,s=40,marker=marker)

plt.savefig(save_name)

plt.clf()

plt.title('bias')
save_name = model_name + 'bias.png'

for i in range(4):
    marker = marker_list[i]
    plt.scatter(bias_pca[i][0],bias_pca[i][1],c='g',alpha=0.5,s=40,marker=marker)
    plt.scatter(bias_pca[i+4][0],bias_pca[i+4][1],c='c',alpha=0.5,s=40,marker=marker)

plt.savefig(save_name)



plt.close(f)

print(checkpoint)
