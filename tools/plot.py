import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def plot_mdmap(matrix,offset,mask):
    path = os.path.join('img_offset')
    k_size = int(offset.size(1)/2)
    
    f = plt.figure(figsize=(6,4))
    cmap = plt.get_cmap('Oranges')
    b,c,d,t = offset.size()

    matrix = matrix.detach().cpu().numpy()
    offset = offset.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    if k_size == 9:
        x_list = [-1,0,1,-1,0,1,-1,0,1]
        y_list = [-1,-1,-1,0,0,0,1,1,1]
    elif k_size == 1:
        x_list = [0]
        y_list = [0]
    for i in range(b):
        for idx in range(d):
            for jdx in range(t):
                plt.matshow(np.max(matrix[i],axis = 0),cmap = cmap)
                plt.colorbar()
                plt.scatter(jdx,idx,color='black',marker='o')
                for c in range(k_size):
                    x = jdx + x_list[c] + offset[i][c][idx][jdx]
                    y = idx + y_list[c] + offset[i][c+k_size][idx][jdx]
                    plt.scatter(x,y,color='blue',marker='o',alpha=mask[i,c,idx,jdx])

                sub_path = os.path.join(path,str(i))
                if not os.path.exists(sub_path):
                    os.makedirs(sub_path)
                plt.savefig(os.path.join(sub_path,str(idx)+'_'+str(jdx)+'.png'))
                plt.clf()
    
    plt.close(f)
    exit()


def plot_map(score_maps, index, model_name):
    # 可视化保存路径
    path = os.path.join('img',model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    
    batch_size = score_maps.shape[0]
    for i in range(batch_size):
        score_map = score_maps[i]
        f = plt.figure(figsize=(6,4))
        plt.matshow(score_map, cmap = plt.cm.cool)
        plt.ylabel("duration")
        plt.xlabel("start time")
        plt.colorbar()
        plt.savefig(os.path.join(path,str(index[i])+'.png'))
        plt.close(f)
        


def plot_pca(video_embeddings,sentence_embeddings,df):

    # 可视化保存路径
    path = os.path.join('img','embedding')
    if not os.path.exists(path):
        os.makedirs(path)
    
    # 读取视频名称
    video_list = df['video']
    length = len(sentence_embeddings)
    data = np.concatenate((sentence_embeddings,video_embeddings),axis=0)
    data_pca = PCA(n_components=2).fit_transform(data)
    max_label = np.max(data_pca,axis=0)
    min_label = np.min(data_pca,axis=0)
    sentence_pca = data_pca[:length]
    video_pca = data_pca[length:]
    # 10个视频画在一张图上
    marker_list = ['.','<','1','8','s','p','*','h','+','x','d','_']
    for index in range(length):

        video_name = video_list[index]
        sentence = sentence_pca[index]
        video = video_pca[index]

        if index == 0:
            f = plt.figure(figsize=(6,4))
            plt.title('video: %s'%(video_name))
            plt.xlim((min_label[0],max_label[0]))
            plt.ylim((min_label[1],max_label[1]))
            save_name = str(index)+'.png'
            pre_video = video_name
            marker = marker_list[0]
            marker_cnt = 0

        if pre_video != video_name:
            '''
            plt.savefig(os.path.join(path,save_name))
            plt.close(f)
            f = plt.figure(figsize=(6,4))
            plt.title('video: %s'%(video_name))
            save_name = str(index)+'.png'
            '''
            marker_cnt += 1
            if marker_cnt == 10:
                marker_cnt = 0
                plt.savefig(os.path.join(path,save_name))
                plt.close(f)
                f = plt.figure(figsize=(6,4))
                plt.xlim((min_label[0],max_label[0]))
                plt.ylim((min_label[1],max_label[1]))
                plt.title('video: %s'%(video_name))
                save_name = str(index)+'.png'
            marker = marker_list[marker_cnt]
            plt.scatter(sentence[0],sentence[1],c='g',alpha=0.5,s=40,marker=marker)
            plt.scatter(video[0],video[1],c='m',alpha=0.5,s=40,marker=marker)
        else:
            plt.scatter(sentence[0],sentence[1],c='g',alpha=0.5,s=40,marker=marker)
            plt.scatter(video[0],video[1],c='m',alpha=0.5,s=40,marker=marker)

        pre_video = video_name

def cIoU(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
    return float(intersection)/union

def plot_similarity(similarity_all,video_lengths,top1,df):
    pre_128 = 0
    pre_256 = 0
    gt_128 = 0
    gt_256 = 0
    too_small = 0
    too_large = 0
    local_max = 0
    small_std = 0
    too_small_iou = 0

    path = os.path.join('img','similarity')
    if not os.path.exists(path):
        os.makedirs(path)

    # 读取GT
    start_segment=df['start_segment']
    end_segment=df['end_segment']

    for (index,similarity) in similarity_all:
        # 视频的长度
        batch_size = similarity.shape[0]
        max_similarity = np.max(similarity)
        for ind in range(batch_size):
            len_video=int(video_lengths[index[ind]])

            break_128=int(np.floor(len_video*2/3))

            gt_start = start_segment[index[ind]]
            gt_end = end_segment[index[ind]]

            # 计算GT和所有滑窗的iou
            start_128 = range(break_128)
            end_128 = range(1,int(break_128+1))
            start_256 = range(int(len_video-break_128))
            end_256 = range(1,int(len_video-break_128+1))
            iou_list=[]
            for start,end in zip(start_128,end_128):
                iou = cIoU((gt_start,gt_end),(start*128,end*128))
                iou_list.append(iou)
            for start,end in zip(start_256,end_256):
                iou = cIoU((gt_start,gt_end),(start*256,end*256))
                iou_list.append(iou)
            iou_array = np.array(iou_list)
            iou_max = np.argmax(iou_array)

            # 统计128窗口的得分分布情况
            # 标准差
            score = similarity[ind,:len_video]
            score_std = np.std(score)
            if score_std < 0.025 :
                small_std += 1

            # 局部最大值
            score_argmax = np.argmax(score)
            score_max = np.max(score)
            local_max_cnt = 0
            for i in range(len_video):
                if i == score_argmax:
                    continue
                elif i == 0 :
                    if score[0] > score[1] and score[0] > score_max-0.01:
                        local_max_cnt += 1
                elif i == len_video-1:
                    if score[len_video-1] > score[len_video-2] and score[len_video-1] > score_max-0.01:
                        local_max_cnt += 1
                else:
                    if score[i] > score[i-1] and score[i] > score[i+1] and score[i] > score_max-0.01:
                        local_max_cnt += 1
            if local_max_cnt > 0:
                local_max += 1

            # 起始帧
            rank1_start=top1[index[ind]]
            if (rank1_start<break_128):
                # 128的滑窗
                pre_128 += 1
                rank1_start_seg =rank1_start*128
                rank1_end_seg = rank1_start_seg+128
            else:
                # 256的滑窗
                pre_256 += 1
                rank1_start_seg =(rank1_start-break_128)*256
                rank1_end_seg = rank1_start_seg+256

            gt_start_128 = gt_start/128.0
            gt_end_128 = gt_end/128.0
            gt_start_256 = gt_start/256.0 + break_128
            gt_end_256 = gt_end/256.0 + break_128
            if iou_max < break_128 :
                gt_128 += 1
                is_gt_128 = True
            else:
                gt_256 += 1
                is_gt_128 = False
            
            iou = cIoU((gt_start,gt_end),(rank1_start_seg,rank1_end_seg))
            if iou_max < break_128:
                if rank1_start >= break_128:
                    too_large += 1
            else:
                if rank1_start < break_128:
                    too_small += 1
                    if iou > 0:
                        too_small_iou +=1

            f = plt.figure(figsize=(6,4))
            # 绘制分界线
            plt.plot([break_128,break_128],[0,max_similarity],linestyle=":",color='gray')
            plt.plot([len_video,len_video],[0,max_similarity],linestyle=":",color='gray')
            # 绘制预测区域
            plt.plot([rank1_start,rank1_start+1],[max_similarity*0.6,max_similarity*0.6],linewidth=4,color='darkred')
            # 绘制GT区域
            if is_gt_128:
                plt.plot([gt_start_128,gt_end_128],[max_similarity*0.4,max_similarity*0.4],linewidth=4,color='orange',marker='o')
                plt.plot([gt_start_256,gt_end_256],[max_similarity*0.4,max_similarity*0.4],linewidth=4,color='orange')
            else:
                plt.plot([gt_start_128,gt_end_128],[max_similarity*0.4,max_similarity*0.4],linewidth=4,color='orange')
                plt.plot([gt_start_256,gt_end_256],[max_similarity*0.4,max_similarity*0.4],linewidth=4,color='orange',marker='o')
            # 绘制得分曲线
            x = range(len_video)
            y = score[x]
            plt.plot(x,y,'g--',marker='o')

            plt.xlabel('len: %d,  break_128: %d,  rank1: %d,  std: %0.3f,  local_max: %d'
                        %(len_video,break_128,rank1_start,score_std,local_max_cnt))
            plt.title('GT: %d-%d,  Pre: %d-%d,  IOU: %0.2f'%(gt_start,gt_end,rank1_start_seg,rank1_end_seg,iou))
            plt.savefig(os.path.join(path,str(index[ind])+'.png'))
            plt.close(f)

    print('pre_128: %d,  pre_256: %d,  gt_128: %d,  gt_256: %d,  too_small: %d,  too_large: %d,  small_std: %d,  local_max: %d,  too_small_iou: %d'
        %(pre_128,pre_256,gt_128,gt_256,too_small,too_large,small_std,local_max,too_small_iou))

def plot_sentence(sentence_embeddings,df):

    path = os.path.join('img','sentence')
    if not os.path.exists(path):
        os.makedirs(path)

    text = df['description']
    length = len(text)
    x = range(len(sentence_embeddings[0]))
    for index in range(length):
        f = plt.figure(figsize=(6,4))
        plt.title(text[index])
        plt.scatter(x,sentence_embeddings[index])
        plt.savefig(os.path.join(path,str(index)+'.png'))
        plt.close(f)

def plot_video(video_embeddings,video_lengths,df):
    path = os.path.join('img','video')
    if not os.path.exists(path):
        os.makedirs(path)

    video = df['video']
    length = len(video)
    x = range(video_embeddings.shape[1])

    for index in range(length):
        f = plt.figure(figsize=(6,4))
        plt.title('video: %s'%video[index])
        #for i in range(video_lengths[index]):
        #    plt.plot(x,video_embeddings[index,i,:])
        plt.scatter(x,video_embeddings[index,:])
        plt.savefig(os.path.join(path,str(index)+'.png'))
        plt.close(f)
