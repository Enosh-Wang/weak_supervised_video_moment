import numpy as np
import torch
from tools.util import iou_with_anchors
import time
def get_lambda(iters, max_iter, continuation_func): # 原版函数应当是适用于 max_iter=20 的情况
<<<<<<< HEAD
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

def get_video_score_nms(scores, valid_num, lam, iou_maps, p_ind):
    
    # score [b,d*t]
    # 对得分从大到小排序，并筛掉非法的部分
    orders = torch.argsort(scores, dim=1, descending=True)[:,:valid_num].detach().cpu().numpy()

    v_score = []
    for i in range(scores.size(0)):
    # 取合法部分的均值作为视频整体的得分
        order = orders[i]
        score = scores[i,order]
        if i != p_ind: 
            v_score.append(torch.sum(score))
        else:
            # 单独对正包做处理
            iou_map = iou_maps[order[0],order]
            inds = np.where(iou_map >= lam)[0]

            v_score.append(torch.mean(score[inds]))

            neg_inds = np.where(iou_map < lam)[0]
            if len(neg_inds) == 0:
                neg_score = torch.tensor(0.).cuda()
            else:
                neg_score = torch.mean(score[neg_inds])

    return torch.stack(v_score), neg_score
    
def get_video_score_nms_all(score, lam, iou_maps, orders):
=======
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

def get_video_score_nms( scores, lam, iou_maps, orders):
    
    #score [b,d*t] mask [d*t]
	video_score = []
	neg_score = []
	p_loss = []
	n_loss = []
	#orders = torch.argsort(scores, dim=1, descending=True)[:,:valid_num].detach().cpu().numpy()
	
	for i in range(scores.size(0)):
		order = orders[i]
			
		index = order[0] # score [d*t]

		iou_map = iou_maps[index,order]
		inds = np.where(iou_map >= lam)[0]

		temp = scores[i,order[inds]]

		max_score = temp[0]
		max_score = max_score.expand_as(temp)
		
		p_loss.append((max_score-temp).mean())

		# weight = torch.Tensor(iou_map[inds]).cuda()
		# weight = 2 - weight
		# label = (weight > 0.7).float() - (weight > 0.9).float()
		# label = label*0.2 + 1
		video_score.append(torch.mean(temp))

		neg_inds = np.where(iou_map < lam)[0]
		if len(neg_inds) == 0:
			neg_score.append(torch.tensor(0.).cuda()) 
			n_loss.append(torch.tensor(0.).cuda())
		else:
			neg_temp = scores[i,order[neg_inds]]
			neg_score.append(torch.mean(neg_temp))	
			min_score = neg_temp[-1]
			min_score = min_score.expand_as(neg_temp)
			n_loss.append((neg_temp-min_score).mean())


	return torch.stack(video_score),torch.stack(neg_score),torch.stack(p_loss),torch.stack(n_loss)
	
def get_video_score_nms_all(scores, lam, iou_maps, orders):
>>>>>>> b39d49fd1a7336ed7eb4478e38a14def76ad2247
    # 完整版CMIL
    #score [b,d*t] mask [d*t]
    video_score = []
    #orders = torch.argsort(score, dim=1, descending=True)[:,:valid_num].detach().cpu().numpy()

    for i in range(score.size(0)):
        sub_score = []
        order = orders[i]
        score = score[i]
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






