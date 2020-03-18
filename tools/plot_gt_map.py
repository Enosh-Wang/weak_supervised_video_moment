import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def get_match_map(temporal_scale):
    match_map = []
    temporal_gap = 1. / temporal_scale
    for idx in range(temporal_scale):
        tmp_match_window = []
        xmin = temporal_gap * idx
        for jdx in range(1, temporal_scale + 1):
            xmax = xmin + temporal_gap * jdx
            tmp_match_window.append([xmin, xmax])
        match_map.append(tmp_match_window)
    match_map = np.array(match_map)  # 100x100x2
    match_map = np.transpose(match_map, [1, 0, 2])  # [0,1] [1,2] [2,3].....[99,100]
    match_map = np.reshape(match_map, [-1, 2])  # [0,2] [1,3] [2,4].....[99,101]   # duration x start
    return match_map  # duration is same in row, start is same in col

def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    # print inter_len,union_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard

def plot_map(score_map, index):
    # 可视化保存路径
    path = os.path.join('img','gt_map_spare')
    if not os.path.exists(path):
        os.makedirs(path)

    f = plt.figure(figsize=(6,4))
    plt.matshow(score_map, cmap = plt.cm.cool)
    plt.ylabel("duration")
    plt.xlabel("start time")
    plt.colorbar()
    plt.savefig(os.path.join(path,str(index)+'.png'))
    plt.close(f)

def get_mask(tscale):

    bm_mask = []
    # 遍历每行，逐行计算掩码，逐行在末尾增加0
    for idx in range(tscale):
        mask_vector = [1 for i in range(tscale - idx)
                    ] + [0 for i in range(idx)]
        bm_mask.append(mask_vector)
    bm_mask = np.array(bm_mask, dtype=np.int)
    return bm_mask

def get_mask_spare(tscale):
         
    bm_mask = []
    # 遍历每行，逐行计算掩码，逐行在末尾增加0
    for duration in range(1, tscale+1):
        mask_vector = []
        k = np.ceil(np.log2(duration/6))
        s = 2**(k-1)
        if k == 1:
            s2 = 0
        else:
            s2 = 2**(k+2)-1

        for start in range(tscale-duration+1):

            if np.mod(start,s) == 0 and np.mod(start+duration-s2,s) == 0:
                mask_vector.append(1)
            else:
                mask_vector.append(0)
                
        mask_vector += [0 for i in range(duration-1)]
        bm_mask.append(mask_vector)
    bm_mask = np.array(bm_mask, dtype=np.int)
    return bm_mask

if __name__ == '__main__':
    path = "/home/share/wangyunxiao/Charades/caption/charades_test.csv"
    df = pd.read_csv(open(path,'rb'))
    start_time = df['start_time']
    end_time = df['end_time']
    duration = df['duration']
    temporal_scale = 20
    match_map = get_match_map(temporal_scale)
    mask = get_mask_spare(temporal_scale)
    plot_map(mask,10)
    exit()
    for i in range(5):
        # 把时间戳转换成百分比
        tmp_start = max(min(1, start_time[i] / duration[i]), 0)
        tmp_end = max(min(1, end_time[i] / duration[i]), 0)

        tmp_gt_iou_map = iou_with_anchors(match_map[:, 0], match_map[:, 1], tmp_start, tmp_end)
        tmp_gt_iou_map = np.reshape(tmp_gt_iou_map, [temporal_scale, temporal_scale])
        plot_map(tmp_gt_iou_map*mask,i)



