import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from util import get_match_map, iou_with_anchors, get_mask, get_mask_spare


def plot_map(score_map, index, duration_start, duration_end):
    # 可视化保存路径
    path = os.path.join('img','gt')
    if not os.path.exists(path):
        os.makedirs(path)

    f = plt.figure(figsize=(6,4))
    cmap = plt.get_cmap('Oranges')
    plt.matshow(score_map, cmap = cmap)
    plt.ylabel("duration")
    plt.xlabel("start time")
    
    plt.colorbar()
    plt.savefig(os.path.join(path,str(index)+'.png'))
    plt.close(f)


if __name__ == '__main__':
    path = "/home/share/wangyunxiao/Charades/caption/charades_test.csv"
    df = pd.read_csv(open(path,'rb'))

    start_time = df['start_time']
    end_time = df['end_time']
    duration = df['duration']

    tscale = 20
    start_ratio = 0.05
    end_ratio = 0.4

    duration_start = int(tscale*start_ratio)
    duration_end = int(tscale*(1-end_ratio))

    match_map = get_match_map(tscale,start_ratio,end_ratio)
    mask = get_mask(tscale,start_ratio,end_ratio)

    cnt = 0
    count = 0
    total_num = len(df)

    for i in range(1):
        # 把时间戳转换成百分比
        tmp_start = max(min(1, start_time[i] / duration[i]), 0)
        tmp_end = max(min(1, end_time[i] / duration[i]), 0)

        tmp_gt_iou_map = iou_with_anchors(match_map[:, 0], match_map[:, 1], tmp_start, tmp_end)
        
        # if max(tmp_gt_iou_map) > 0.7:
        #     cnt += 1
        # b_map = tmp_gt_iou_map > 0.3
        # count += sum(b_map)
        # b_map = np.reshape(b_map, [-1, tscale])
        # plot_map(b_map*mask,i,duration_start, duration_end)

        tmp_gt_iou_map = np.reshape(tmp_gt_iou_map, [-1, tscale])
        b_map = tmp_gt_iou_map > 0.8
        plot_map(tmp_gt_iou_map*mask*b_map,i,duration_start, duration_end)

    # print('ReCall IOU=0.7:',cnt/total_num)
    # print('IOU=0.7 avg_num:',count/total_num)



