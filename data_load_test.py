import os
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, LSTM
from matplotlib import pyplot as plt

tag = [[0], [1]]
shape = (1, 5, 17 * 2)

load_path = './source/walk'
size = 19
val_tag = 0


def change_shape(arr=None):
    if arr is None:
        arr = []
    arr_changed = np.ones((5, 17 * 2))
    for i in range(0, 17):
        for j in range(0, 2):
            arr_changed[0][i * 2 + j] = arr[0][i][j]
            arr_changed[1][i * 2 + j] = arr[1][i][j]
            arr_changed[2][i * 2 + j] = arr[2][i][j]
            arr_changed[3][i * 2 + j] = arr[3][i][j]
            arr_changed[4][i * 2 + j] = arr[4][i][j]
    return arr_changed


x_final = []
x_final1 = []
x_final2 = []
y_final = []
y_final1 = []
y_final2 = []
for i in range(1, size + 1):
    path = load_path + '/' + i.__str__() + '/0.npy'
    x = np.load(path)
    x = change_shape(x)
    x = np.reshape(x, (1, 5, 17 * 2))
    y = tag[val_tag]
    x_final1 = x_final2
    y_final1 = y_final2
    for j in range(1, 500):
        path = load_path + '/' + i.__str__() + '/' + j.__str__() + '.npy'
        if not os.path.exists(path):
            break
        read_data_tem = []
        try:
            read_data_tem = np.load(path)
            read_data_tem = change_shape(read_data_tem)
        except Exception as e:
            print('the SIZE in ' + load_path + ' is out of range!', e.args)
        read_data_tem = np.reshape(read_data_tem, (1, 5, 17 * 2))
        x = np.append(x, read_data_tem, axis=0)
        y = np.append(y, tag[val_tag], axis=0)
        x_final2 = x
        y_final2 = y
    if i == 2:
        x_final = np.append(x_final1, x_final2, axis=0)
        y_final = np.append(y_final1, y_final2, axis=0)
    if i > 2:
        x_final = np.append(x_final, x, axis=0)
        y_final = np.append(y_final, y, axis=0)

y_final = np.reshape(y_final, (len(y_final), 1))
num = math.ceil(y_final.shape[0] * 0.8)
print(num)
y1 = y_final[0:num]
y2 = y_final[:-num]
print(x_final.shape)
print(y_final.shape)
print(y1.shape)
print(y2.shape)
