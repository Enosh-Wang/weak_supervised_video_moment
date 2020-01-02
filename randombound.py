import numpy as np
import pandas as pd
import os
import scipy.io as sio
import skimage.measure as scikit

def cIoU(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
    return float(intersection)/union


csv_path = '/home/share/wangyunxiao/Charades/charades_precomp/Caption'
df = pd.read_csv(os.path.join(csv_path,'charades_test.csv'))

start_segment=df['start_segment']
end_segment=df['end_segment']
video_name = df['video']

length = len(df)
num03 = 0
num05 = 0
num07 = 0

R5IOU5=0
R5IOU7=0
R5IOU3=0
R10IOU3=0
R10IOU5=0
R10IOU7=0
for i in range(length):
    gt_start = start_segment[i]
    gt_end = end_segment[i]
    video_feat_file="/home/share/wangyunxiao/Charades/charades_precomp/c3d_features/"+str(video_name[i])+".mat"
    video_feat_mat = sio.loadmat(video_feat_file)
    video_feat=video_feat_mat['feature']
    video_feat1=scikit.block_reduce(video_feat, block_size=(8, 1), func=np.mean)
    video_feat2=scikit.block_reduce(video_feat, block_size=(16, 1), func=np.mean)

    range_128 = range(len(video_feat1))
    range_256 = range(len(video_feat2))

    iou_list = []
    for i in range_128:
        start = i*128
        end = (i+1)*128
        iou = cIoU((start,end),(gt_start,gt_end))
        iou_list.append(iou)

    for i in range_256:
        start = i*256
        end = (i+1)*256
        iou = cIoU((start,end),(gt_start,gt_end))
        iou_list.append(iou)

    iou_array = np.array(iou_list)
    print(iou_array)
    iou_max = np.random.choice(iou_array, 1, replace=False)
    #np.random.shuffle(iou_array)
    #iou_max = iou_array[0]

    if iou_max>=0.3:
        num03+=1
    if iou_max>=0.5:
        num05+=1
    if iou_max>=0.7:
        num07+=1

print(float(num03)/length)
print(float(num05)/length)
print(float(num07)/length)
for j1 in range(5):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.5:
               R5IOU5+=1
               break
			   
        for j1 in range(5):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.7:
               R5IOU7+=1
               break
			   
        for j1 in range(5):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.3:
               R5IOU3+=1
               break
			   
        for j1 in range(10):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.5:
               R10IOU5+=1
               break
			   
        for j1 in range(10):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.7:
               R10IOU7+=1
               break
			   
        for j1 in range(10):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.3:
               R10IOU3+=1
               break    
			    	
			   
	
	
	############################

    # Compute metrics
    R1IoU05=correct_num05
    R1IoU07=correct_num07
    R1IoU03=correct_num03
    total_length=attn_index.shape[0]
    #print('total length',total_length)
    print("R@1 IoU0.3: %f" %(R1IoU03/float(total_length)))
    print("R@5 IoU0.3: %f" %(R5IOU3/float(total_length)))
    print("R@10 IoU0.3: %f" %(R10IOU3/float(total_length)))
	
    print("R@1 IoU0.5: %f" %(R1IoU05/float(total_length)))
    print("R@5 IoU0.5: %f" %(R5IOU5/float(total_length)))
    print("R@10 IoU0.5: %f" %(R10IOU5/float(total_length)))
	
    print("R@1 IoU0.7: %f" %(R1IoU07/float(total_length)))
    print("R@5 IoU0.7: %f" %(R5IOU7/float(total_length)))
    print("R@10 IoU0.7: %f" %(R10IOU7/float(total_length)))