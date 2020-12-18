
python main.py --model_name test

--resume checkpoint的路径，恢复训练时用
更换数据集要更改以下参数：
--dataset
--video_dim
--margin

CUDA_VISIBLE_DEVICES=3 nohup python main.py --model_name ActivityNet --dataset ActivityNet --temporal_scale 100 --video_dim 500 --batch_size 4 --num_epochs 10 --end_ratio 0.2 --num_sample 32 --global_margin 0.4 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --model_name tacos --dataset TACoS --temporal_scale 128 --video_dim 4096 --batch_size 4 --num_epochs 10 --end_ratio 0.2 --num_sample 32 --global_margin 0.4 &