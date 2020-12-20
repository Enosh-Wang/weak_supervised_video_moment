import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from util import get_match_map, iou_with_anchors, get_mask, get_mask_spare
from tqdm import tqdm

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
    #train_path = "/home/share/wangyunxiao/ActivityNet/caption/activitynet_train.csv"
    train_path = "/home/share/wangyunxiao/TACoS/caption/tacos_train.csv"
    df = pd.read_csv(open(train_path,'rb'))

    start_time = df['start_time']
    end_time = df['end_time']
    duration = df['duration']

    tscale = 128
    start_ratio = 0
    end_ratio = 0

    duration_start = int(tscale*start_ratio)
    duration_end = int(tscale*(1-end_ratio))

    match_map = get_match_map(tscale,start_ratio,end_ratio)
    mask = get_mask(tscale,start_ratio,end_ratio)

    cnt = 0
    count = 0
    total_num = len(df)
    cnt_map = np.zeros(tscale**2)
    for i in tqdm(range(total_num)):
        # 把时间戳转换成百分比
        tmp_start = max(min(1, start_time[i] / duration[i]), 0)
        tmp_end = max(min(1, end_time[i] / duration[i]), 0)

        tmp_gt_iou_map = iou_with_anchors(match_map[:, 0], match_map[:, 1], tmp_start, tmp_end)
        b_map = tmp_gt_iou_map > 0.7
        cnt_map += b_map
        # if max(tmp_gt_iou_map) > 0.7:
        #     cnt += 1
        # b_map = tmp_gt_iou_map > 0.7
        # count += sum(b_map)
        # b_map = np.reshape(b_map, [-1, tscale])
        # plot_map(b_map*mask,i,duration_start, duration_end)

        # tmp_gt_iou_map = np.reshape(tmp_gt_iou_map, [-1, tscale])
        # b_map = tmp_gt_iou_map > 0.8
        # plot_map(tmp_gt_iou_map*mask*b_map,i,duration_start, duration_end)

    # print('ReCall IOU=0.7:',cnt/total_num)
    # print('IOU=0.7 avg_num:',count/total_num)
    cnt_map = np.reshape(cnt_map, [-1, tscale])
    print(np.argmax(cnt_map))
    plot_map(cnt_map*mask,i,duration_start, duration_end)


