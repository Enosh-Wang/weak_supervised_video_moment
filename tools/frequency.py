import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from util import get_match_map, iou_with_anchors, get_mask, get_mask_spare
from tqdm import tqdm

def cIoU(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
    return float(intersection)/union

if __name__ == '__main__':
    # val_path = "/home/share/wangyunxiao/Charades/caption/charades_val.csv"
    # test_path = "/home/share/wangyunxiao/Charades/caption/charades_test.csv"
    val_path = "/home/share/wangyunxiao/ActivityNet/caption/activitynet_val.csv"
    test_path = "/home/share/wangyunxiao/ActivityNet/caption/activitynet_test.csv"
    df = pd.read_csv(open(test_path,'rb'))
    # df = pd.read_csv(open(val_path,'rb'))

    start_time = df['start_time']
    end_time = df['end_time']
    duration = df['duration']

    cnt_7 = 0
    cnt_5 = 0
    cnt_3 = 0
    total_num = len(df)

    for i in tqdm(range(total_num)):
        # 把时间戳转换成百分比
        tmp_start = max(min(1, start_time[i] / duration[i]), 0)
        tmp_end = max(min(1, end_time[i] / duration[i]), 0)

        iou = cIoU((tmp_start, tmp_end),(0.22, 0.92)) # (0, 0.35)
        if iou > 0.7:
            cnt_7 += 1
        if iou > 0.5:
            cnt_5 += 1
        if iou > 0.3:
            cnt_3 += 1
    
    print("R@1 IoU0.7: %f" %(cnt_7/float(total_num)))
    print("R@1 IoU0.5: %f" %(cnt_5/float(total_num)))
    print("R@1 IoU0.3: %f" %(cnt_3/float(total_num)))



