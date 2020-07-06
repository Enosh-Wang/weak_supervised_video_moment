import numpy as np
import torch
from tools.util import iou_with_anchors
import time
def get_lambda(iters, max_iter, continuation_func): # 原版函数应当是适用于 max_iter=20 的情况
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
		lam = np.exp((iters-max_iter)/4) #2000
	return lam

def get_video_score_nms(scores, lam, iou_maps, orders):
    
    #score [b,d*t] mask [d*t]
	video_score = []
	loss = []
	#orders = torch.argsort(scores, dim=1, descending=True)[:,:valid_num].detach().cpu().numpy()

	for i in range(scores.size(0)):
		order = orders[i]
			
		index = order[0] # score [d*t]

		iou_map = iou_maps[index,order]
		inds = np.where(iou_map >= lam)[0]
		
		temp = scores[i,order[inds]]
		# top = temp[0]
		# top = top.expand_as(temp)
		# loss.append(torch.sum((top - temp)*torch.Tensor(iou_map[inds]).cuda()))


		sub_score = torch.sum(temp)/inds.size

		video_score.append(sub_score)

	return torch.stack(video_score)#torch.stack(loss).mean()

def get_video_score_nms_all(scores, lam, iou_maps, orders):
    # 完整版CMIL
    #score [b,d*t] mask [d*t]
	video_score = []
	#orders = torch.argsort(scores, dim=1, descending=True)[:,:valid_num].detach().cpu().numpy()

	for i in range(scores.size(0)):
		sub_score = []
		order = orders[i]
		score = scores[i]
		num = 0
		while order.size > 0 and num < 10:
			num += 1
			index = order[0] # score [d*t]
			
			iou_map = iou_maps[index]
			inds = np.where(iou_map[order] >= lam)[0]
			
			temp = score[order[inds]]
			sub_score.append( torch.sum(temp)/inds.size )

			order = np.delete(order,inds)
			# inds = np.where(iou_map[order] < lam)[0]
			# order = order[inds]

		video_score.append(torch.max(torch.stack(sub_score)))

	return torch.stack(video_score)

	





