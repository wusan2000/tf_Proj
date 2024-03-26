# 介绍
* 使用tf_DataGenerater-main.py中的DataGenerater_humanpose_20220703_XXX.py得到训练数据（一个视频按一定时间间隔截取一帧帧图像，使用openpose得到人体的数据点，将每一帧的人体结构的数据存入.npy中，每一个图片得到的这些npy文件存在一个文件夹里作为训练数据。
* 使用LSTM_XXX.py训练检测视频5帧是否有跌倒情况的网络。
* 使用GUI_XXX.py进行实时的摔倒监控。
