import os
import numpy as np
from scipy.ndimage import gaussian_filter

def label_frame_by_threshold(score, bw=None, thresh=list([0.05])):

    rst = []
    cls_score = score if bw is None else gaussian_filter(score, bw)
    for th in thresh:
        rst.append((cls_score > th,cls_score))

    return rst

def build_box_by_search(frm_label_lst, tol, min=1):
    boxes = []
    for frm_labels, frm_scores in frm_label_lst:
        # 帧数
        length = len(frm_labels)
        # 依次后项减去前项
        diff = np.empty(length+1)
        diff[1:-1] = frm_labels[1:].astype(int) - frm_labels[:-1].astype(int)
        diff[0] = float(frm_labels[0])
        diff[length] = 0 - float(frm_labels[-1])

        # 逐项累加
        cs = np.cumsum(1 - frm_labels)
        offset = np.arange(0, length, 1)

        # 起止点
        up = np.nonzero(diff == 1)[0]
        down = np.nonzero(diff == -1)[0]

        assert len(up) == len(down), "{} != {}".format(len(up), len(down))
        # tol 阈值列表
        for i, t in enumerate(tol):

            signal = cs - t * offset

            for x in range(len(up)):
                s = signal[up[x]]

                for y in range(x + 1, len(up)):
                    if y < len(down) and signal[up[y]] > s:
                        boxes.append([up[x], down[y-1]+1, sum(frm_scores[up[x]:down[y-1]+1])])
                        break
                else:
                    boxes.append([up[x], down[-1] + 1, sum(frm_scores[up[x]:down[-1] + 1])])

            for x in range(len(down) - 1, -1, -1):
                s = signal[down[x]] if down[x] < length else signal[-1] - t
                for y in range(x - 1, -1, -1):
                    if y >= 0 and signal[down[y]] < s:
                        boxes.append([up[y+1], down[x] + 1, sum(frm_scores[up[y+1]:down[x] + 1])])
                        break
                else:
                    boxes.append([up[0], down[x] + 1, sum(frm_scores[0:down[x]+1 + 1])])

    return boxes

def temporal_nms(bboxes, thresh):
    """
    One-dimensional non-maximal suppression
    :param bboxes: [[st, ed, score, ...], ...]
    :param thresh:
    :return:
    """
    t1 = bboxes[:, 0]
    t2 = bboxes[:, 1]
    scores = bboxes[:, 2]

    durations = t2 - t1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        tt1 = np.maximum(t1[i], t1[order[1:]])
        tt2 = np.minimum(t2[i], t2[order[1:]])
        intersection = tt2 - tt1
        IoU = intersection / (durations[i] + durations[order[1:]] - intersection).astype(float)

        inds = np.where(IoU <= thresh)[0]
        order = order[inds + 1]

    return bboxes[keep, :]

def gen_prop1(scores,index):

    size = scores.shape
    batch_size = size[0]
    postive = []
    batch_result = {}
    step = 1/size[2]
    for i in range(batch_size):
        score = scores[i,i,:]
        postive.append(score)

        #frm_duration = len(scores)
        labels = label_frame_by_threshold(score, bw=3, thresh=[0.01, 0.05, 0.1, .15, 0.25, .4, .5, .6, .7, .8, .9, .95, ])

        bboxes = build_box_by_search(labels, tol=[0.05, .1, .2, .3, .4, .5, .6, 0.8, 1.0])

        # print len(bboxes)
        bboxes = np.asarray(bboxes)
        #bboxes = temporal_nms(bboxes, 0.9)


        bboxes[:,0] = bboxes[:,0]*step
        bboxes[:,1] = (bboxes[:,1]+1)*step
        batch_result[index[i]] = bboxes
        #pr_box = [(x[0] / float(frm_duration) * v.duration, x[1] / float(frm_duration) * v.duration) for x in bboxes]

        # filter out too short proposals
        #pr_box = list(filter(lambda b: b[1] - b[0] > args.minimum_len, pr_box))
    return np.stack(postive),batch_result#pr_box

def gen_prop(score,index):

    size = len(score)
    
    batch_result = {}
    step = 1/size

    labels = label_frame_by_threshold(score, bw=3, thresh=[0.01, 0.05, 0.1, .15, 0.25, .4, .5, .6, .7, .8, .9, .95, ])

    bboxes = build_box_by_search(labels, tol=[0.05, .1, .2, .3, .4, .5, .6, 0.8, 1.0])

    bboxes = np.asarray(bboxes)

    bboxes[:,0] = bboxes[:,0]*step
    bboxes[:,1] = (bboxes[:,1]+1)*step
    batch_result[index[0]] = bboxes

    return batch_result#pr_box