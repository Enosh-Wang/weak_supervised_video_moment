import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from util import get_match_map, iou_with_anchors, get_mask_spare, get_mask
import seaborn as sns
def plot_map(score_map, mask, index, duration_start, duration_end):
    # 可视化保存路径
    path = os.path.join('img','match_map_ratio')
    if not os.path.exists(path):
        os.makedirs(path)

    f = plt.figure(figsize=(12,8))
    sns.heatmap(score_map,annot=True,mask=mask,cmap='Oranges')
    plt.ylabel("duration")
    plt.xlabel("start time")
    plt.savefig(os.path.join(path,str(index)+'.png'))
    plt.close(f)

if __name__ == '__main__':

    tscale = 20
    start_ratio = 0
    end_ratio = 0
    match_map = get_match_map(tscale, start_ratio, end_ratio)
    mask = 1-get_mask(tscale, start_ratio, end_ratio)
    index = 0

    duration_start = int(tscale*start_ratio)
    duration_end = int(tscale*(1-end_ratio))

    sns.set(font_scale=0.7)

    for idx in range(duration_start, duration_end):
        for jdx in range(tscale):
            # 起止点的索引
            start_index = jdx
            end_index = start_index + idx+1
            # 如果是上一步中选定的起止点，则进一步验证置信度
            if end_index <= tscale :
                # proposal的坐标
                xmin = start_index/tscale
                xmax = end_index/tscale
                gt_map = iou_with_anchors(match_map[:, 0], match_map[:, 1], xmin, xmax)
                gt_map = np.reshape(gt_map, [-1, tscale])
                plot_map(gt_map,mask,index,duration_start,duration_end)
                index += 1


