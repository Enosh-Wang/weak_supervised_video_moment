import numpy as np
import pandas as pd
import os
import scipy.io as sio
import skimage.measure as scikit

def cIoU(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
    return float(intersection)/union


csv_path = '/home/share/wangyunxiao/Charades/charades_precomp/Caption'
df = pd.read_csv(os.path.join(csv_path,'charades_test.csv'))

start_segment=df['start_segment']
end_segment=df['end_segment']
video_name = df['video']

length = len(df)
num03 = 0
num05 = 0
num07 = 0

for i in range(length):
    gt_start = start_segment[i]
    gt_end = end_segment[i]
    video_feat_file="/home/share/wangyunxiao/Charades/charades_precomp/c3d_features/"+str(video_name[i])+".mat"
    video_feat_mat = sio.loadmat(video_feat_file)
    video_feat=video_feat_mat['feature']
    video_feat1=scikit.block_reduce(video_feat, block_size=(8, 1), func=np.mean)
    video_feat2=scikit.block_reduce(video_feat, block_size=(16, 1), func=np.mean)

    range_128 = range(len(video_feat1))
    range_256 = range(len(video_feat2))

    iou_list = []
    for i in range_128:
        start = i*128
        end = (i+1)*128
        iou = cIoU((start,end),(gt_start,gt_end))
        iou_list.append(iou)

    for i in range_256:
        start = i*256
        end = (i+1)*256
        iou = cIoU((start,end),(gt_start,gt_end))
        iou_list.append(iou)

    iou_array = np.array(iou_list)
    print(iou_array)
    iou_max = np.random.choice(iou_array, 1, replace=False)
    #np.random.shuffle(iou_array)
    #iou_max = iou_array[0]

    if iou_max>=0.3:
        num03+=1
    if iou_max>=0.5:
        num05+=1
    if iou_max>=0.7:
        num07+=1

print(float(num03)/length)
print(float(num05)/length)
print(float(num07)/length)