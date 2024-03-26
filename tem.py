# from sklearn.preprocessing import MinMaxScaler
# import numpy as np
#
# data = np.array([[4, 2, 3],
#                  [1, 5, 6]])
#
# # 手动归一化
# feature_range = [0, 1]  # 要映射的区间
# print(data.min(axis=0))
# print(data.max(axis=0))
# x_std = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
# x_scaled = x_std * (feature_range[1] - feature_range[0]) + feature_range[0]
# print('手动归一化结果：\n{}'.format(x_scaled))
#
# # 自动归一化
# scaler = MinMaxScaler()
# print('自动归一化结果:\n{}'.format(scaler.fit_transform(data)))

import os
path = "./openvino"
for _ in os.listdir(path):  # 文件、文件夹名字
    print(_)
count = len(os.listdir(path))  # 数量
print(count)