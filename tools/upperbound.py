import numpy as np
import pandas as pd
import os
from util import iou_with_anchors,get_window_list

# csv_path = '/home/share/wangyunxiao/Charades/caption'
# df = pd.read_csv(os.path.join(csv_path,'charades_test.csv'))
# csv_path = '/home/share/wangyunxiao/ActivityNet/caption'
# df = pd.read_csv(os.path.join(csv_path,'activitynet_test.csv'))
csv_path = '/home/share/wangyunxiao/TACoS/caption'
df = pd.read_csv(os.path.join(csv_path,'tacos_test.csv'))


start_time_list = df['start_time']
end_time_list = df['end_time']
duration_list = df['duration']
length = len(df)

for i in range(60,120):
    window_list = get_window_list(i,3,1,255)

    num03 = 0
    num05 = 0
    num07 = 0

    for index in range(length):
        duration = duration_list[index]
        gt_start=start_time_list[index]/duration
        gt_end=end_time_list[index]/duration

        iou_list = iou_with_anchors(window_list[:,0],window_list[:,1],gt_start,gt_end)
        iou_max = np.max(iou_list)

        if iou_max>=0.3:
            num03+=1
        if iou_max>=0.5:
            num05+=1
        if iou_max>=0.7:
            num07+=1

    print('layer:'+str(i))
    print(float(num03)/length)
    print(float(num05)/length)
    print(float(num07)/length)

