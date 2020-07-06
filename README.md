
python main.py --model_name test

--resume checkpoint的路径，恢复训练时用
更换数据集要更改以下参数：
--dataset
--video_dim
--margin

CUDA_VISIBLE_DEVICES=1 python main.py --model_name ActivityNet --dataset ActivityNet --temporal_scale 100 --video_dim 500 --batch_size 8