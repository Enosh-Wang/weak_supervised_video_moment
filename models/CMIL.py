import numpy as np
import torch
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

def get_video_score_nms_list(scores, lam, iou_maps, p_ind):
    
    # score [b,d*t]
    # 对得分从大到小排序，并筛掉非法的部分
    inds = torch.argmax(scores, dim=1).detach().cpu().numpy()
    batch,length = scores.size()
    length = max(int(length*(1-lam)),1)

    v_score = []
    for i in range(batch):
    # 取合法部分的均值作为视频整体的得分
        ind = inds[i]
        score = scores[i]
        iou_map = iou_maps[ind]
        score_iou = score + torch.tensor(iou_map).cuda()
        _,order = score_iou.sort(dim=0,descending=True)
        score = score[order]
        # temp = list(zip(iou_map,score))
        # temp.sort(key=lambda x: (-x[0],-x[1]))
        # score_sorted = [item[1] for item in temp]
        # score_sorted = torch.stack(score_sorted)
        v_score.append(torch.mean(score[:length]))

    return torch.stack(v_score)#, neg_score

def get_video_score_nms_list1(scores, lam, iou_maps, p_ind):
    
    # score [b,d*t]
    # 对得分从大到小排序，并筛掉非法的部分
    orders = torch.argsort(scores, dim=1, descending=True).detach().cpu().numpy()

    v_score = []
    for i in range(scores.size(0)):
    # 取合法部分的均值作为视频整体的得分
        order = orders[i]
        score = scores[i,order]

        iou_map = iou_maps[order[0],order]
        inds = np.where(iou_map >= lam)[0]

        v_score.append(torch.mean(score[inds]))
        if i == p_ind: 
            neg_inds = np.where(iou_map < lam)[0]
            if len(neg_inds) == 0:
                neg_score = torch.tensor(0.).cuda()
            else:
                neg_score = torch.sum(score[neg_inds])

    return torch.stack(v_score)#, neg_score

def get_video_score_nms(scores, valid_num, lam, iou_maps, p_ind):
    
    # score [b,d*t]
    # 对得分从大到小排序，并筛掉非法的部分
    orders = torch.argsort(scores, dim=1, descending=True)[:,:valid_num].detach().cpu().numpy()

    v_score = []
    for i in range(scores.size(0)):
    # 取合法部分的均值作为视频整体的得分
        order = orders[i]
        score = scores[i,order]

        iou_map = iou_maps[order[0],order]
        inds = np.where(iou_map >= lam)[0]

        v_score.append(torch.mean(score[inds]))
        if i == p_ind: 
            neg_inds = np.where(iou_map < lam)[0]
            if len(neg_inds) == 0:
                neg_score = torch.tensor(0.).cuda()
            else:
                neg_score = torch.sum(score[neg_inds])

    return torch.stack(v_score), neg_score
    
def get_video_score_nms_all(score, lam, iou_maps, orders):
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


if __name__ == "__main__":
    scores = torch.rand(32,25)
    lam = 0.3
    iou_maps = np.random.rand(25,25)
    p_ind = 0
    get_video_score_nms_list(scores, lam, iou_maps, p_ind)
    



