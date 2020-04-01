import os
import numpy as np
import multiprocessing as mp
from ops.sequence_funcs import temporal_nms

# bottom-up generate proposals
print('generating proposals')
pr_dict = {}
pr_score_dict = {}
topk = 1

def label_frame_by_threshold(score, bw=None, thresh=list([0.05])):

    ss = softmax(score)
    rst = []
   
        cls_score = ss[:, cls+1] if bw is None else gaussian_filter(ss[:, cls+1], bw)
        for th in thresh:
            rst.append((cls_score > th, f_score[:, cls+1]))

    return rst

def build_box_by_search(frm_label_lst, tol, min=1):
    boxes = []
    for cls, frm_labels, frm_scores in frm_label_lst:
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
                        boxes.append((up[x], down[y-1]+1, cls, sum(frm_scores[up[x]:down[y-1]+1])))
                        break
                else:
                    boxes.append((up[x], down[-1] + 1, cls, sum(frm_scores[up[x]:down[-1] + 1])))

            for x in range(len(down) - 1, -1, -1):
                s = signal[down[x]] if down[x] < length else signal[-1] - t
                for y in range(x - 1, -1, -1):
                    if y >= 0 and signal[down[y]] < s:
                        boxes.append((up[y+1], down[x] + 1, cls, sum(frm_scores[up[y+1]:down[x] + 1])))
                        break
                else:
                    boxes.append((up[0], down[x] + 1, cls, sum(frm_scores[0:down[x]+1 + 1])))

    return boxes


def gen_prop(scores):

    frm_duration = len(scores)
    topk_labels = label_frame_by_threshold(scores, bw=3, thresh=[0.01, 0.05, 0.1, .15, 0.25, .4, .5, .6, .7, .8, .9, .95, ])

    bboxes = []
    tol_lst = [0.05, .1, .2, .3, .4, .5, .6, 0.8, 1.0]

    bboxes.extend(build_box_by_search(topk_labels, np.array(tol_lst)))

    # print len(bboxes)
    bboxes = temporal_nms(bboxes, 0.9)

    pr_box = [(x[0] / float(frm_duration) * v.duration, x[1] / float(frm_duration) * v.duration) for x in bboxes]

    # filter out too short proposals
    pr_box = list(filter(lambda b: b[1] - b[0] > args.minimum_len, pr_box))
    return v.id, pr_box, [x[3] for x in bboxes]



pool = mp.Pool(processes = 32)
lst = []
handle = [pool.apply_async(gen_prop, args=(x, ), callback=call_back) for x in video_list]
pool.close()
pool.join()
