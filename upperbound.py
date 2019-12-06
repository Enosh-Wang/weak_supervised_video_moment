import numpy as np
import pandas as pd
import os

def cIoU(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
    return float(intersection)/union


csv_path = '/home/share/wangyunxiao/Charades/charades_precomp/Caption'
df = pd.read_csv(os.path.join(csv_path,'charades_test.csv'))

start_segment=df['start_segment']
end_segment=df['end_segment']
length = len(df)
num03 = 0
num05 = 0
num07 = 0

for i in range(length):
    gt_start = start_segment[i]
    gt_end = end_segment[i]
    range_128 = range(int(gt_start/128),int(gt_end/128)+1)
    range_256 = range(int(gt_start/256),int(gt_end/256)+1)

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

    iou_max = np.max(np.array(iou_list))

    if iou_max>=0.3:
        num03+=1
    if iou_max>=0.5:
        num05+=1
    if iou_max>=0.7:
        num07+=1

print(float(num03)/length)
print(float(num05)/length)
print(float(num07)/length)