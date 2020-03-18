import numpy as np
import torch
from tools.plot_gt_map import iou_with_anchors
import time
def get_lambda(iters, max_iter, continuation_func):
	if continuation_func == 'Linear':
		lam = iters/max_iter
	elif continuation_func == 'Plinear':
		lam = min(1, np.ceil(iters/4)/max_iter*4)
	elif continuation_func == 'Sigmoid':
		lam = 1/(1+np.exp(max_iter/2-iters))
	elif continuation_func == 'Log':
		low_bound = 0.01
		lam = (np.log(iters+low_bound)-np.log(low_bound))/(np.log(max_iter+low_bound)-np.log(low_bound))
	elif continuation_func == 'Exp':
		lam = np.exp((iters-max_iter)/4)
	return lam


def get_video_score_nms(scores, lam, tscale, match_map, valid_num):
    
    #score [b,d*t] mask [d*t]
	video_score = []
	orders = torch.argsort(scores, dim=1, descending=True)[:,:valid_num].detach().cpu().numpy()

	for i in range(scores.size(0)):
		sub_score = []
		order = orders[i]
		score = scores[i]
		while order.size > 0:
			
			index = order[0] # score [d*t]
			start_index = index%tscale
			end_index = start_index + index//tscale + 1
			xmin = start_index/tscale
			xmax = end_index/tscale
			
			iou_map = iou_with_anchors(match_map[order, 0],match_map[order, 1],xmin,xmax)
			inds = np.where(iou_map >= lam)[0]
			
			temp = score[order[inds]]
			sub_score.append( torch.sum(temp)/inds.size )

			inds = np.where(iou_map < lam)[0]
			order = order[inds]

		video_score.append(torch.max(torch.stack(sub_score)))

	return torch.stack(video_score)

	





