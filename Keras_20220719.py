import os
import numpy as np
import tensorflow as tf
import math
from tensorflow.keras.layers import Dropout, Flatten, Input, Dense
from matplotlib import pyplot as plt


input_shape = (5, 17 * 2)
num_classes = 2
source_path = './source'
npy_save_path = r'C:\Users\thee\PycharmProjects\tf_Proj_predict20220624\source\DATA_tem\2\data.npy'
timber_path = r'C:\Users\thee\PycharmProjects\tf_Proj_predict20220624\source\DATA_tem\2\timber'
walk_path = r'C:\Users\thee\PycharmProjects\tf_Proj_predict20220624\source\DATA_tem\2\walk'
tag = [[0], [1]]
tag_class = {'timber': 0, 'walk': 1}


# Define a simple sequential model
# return_sequences一般在最后一层为False
def create_model():
    model = tf.keras.Sequential([
        Input(shape=input_shape),
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()),
        Dropout(0.1),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()),
        Dropout(0.1),
        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, clipnorm=1),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'], )
    return model


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


# 读取训练数据

def read_data(load_path='', size=0, val_tag=0):
    """
    读取训练数据函数：
    load_path是数据所在文件夹（timber或者walk）；
    size是文件夹号码最大的数字（比如有0-49文件夹，则size=50）；
    val_tag指的是数据标签（'timber': 0, 'walk': 1）
    """
    x_final = []
    x_final2 = []
    y_final = []
    y_final2 = []
    for i in range(0, size):
        path = load_path + '/' + i.__str__() + '/0.npy'
        x = np.load(path)
        x = change_shape(x)
        x = np.reshape(x, (1, 5, 17 * 2))
        y = tag[val_tag]
        x_final1 = x_final2
        y_final1 = y_final2
        # if not os.path.exists((load_path + '/' + i.__str__())):
        #     print(load_path + '/' + i.__str__())
        #     continue
        for j in range(0, 50000):
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
        if i == 1:
            x_final = x
            y_final = y
        elif i == 2:
            x_final = np.append(x_final1, x_final2, axis=0)
            y_final = np.append(y_final1, y_final2, axis=0)
        elif i > 2:
            x_final = np.append(x_final, x, axis=0)
            y_final = np.append(y_final, y, axis=0)

    y_final = np.reshape(y_final, (len(y_final), 1))
    # print(x_final.shape)
    # print(y_final.shape)

    return x_final, y_final


def main():
    timber_files = os.listdir(timber_path)  # 读入文件夹
    timber_size = len(timber_files)  # 统计文件夹中的文件个数
    walk_files = os.listdir(walk_path)  # 读入文件夹
    walk_size = len(walk_files)  # 统计文件夹中的文件个数
    x1, y1 = read_data(timber_path, timber_size, 0)
    x2, y2 = read_data(walk_path, walk_size, 1)
    # x1, y1 = read_data(timber_path, 10, 0)
    # x2, y2 = read_data(walk_path, 4, 1)
    train_x = np.append(x1, x2, axis=0)
    train_y = np.append(y1, y2, axis=0)
    print('data of timber: {}'.format(x1.shape))
    print('data of walk: {}'.format(x2.shape))
    # 打乱数据
    # seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
    np.random.seed(116)  # 使用相同的seed，保证输入特征和标签一一对应
    np.random.shuffle(train_x)
    np.random.seed(116)
    np.random.shuffle(train_y)
    tf.random.set_seed(116)

    train_x, train_y = np.array(train_x), np.array(train_y)
    # train_x = train_x / 3
    print(train_x)
    print(train_x.shape, train_y.shape)

    # 取出部分数据作为测试集
    num = math.ceil(train_x.shape[0] * 0.8)
    test_x, test_y = train_x[:-num], train_y[:-num]
    train_x, train_y = train_x[:num], train_y[:num]
    # Create a basic model instance
    model = create_model()

    checkpoint_save_path = "./checkpoint/lstm.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)
    # print(model.trainable_variables)
    # file = open('./weights.txt', 'w')
    # for v in model.trainable_variables:
    #     file.write(str(v.name) + '\n')
    #     file.write(str(v.shape) + '\n')
    #     # file.write(str(v.numpy()) + '\n')
    # file.close()

    # model.fit(train_data, train_y, epochs=10)
    history = model.fit(train_x, train_y, batch_size=256, epochs=100, validation_data=(test_x, test_y),
                        validation_freq=1,
                        callbacks=[cp_callback])

    # Display the model's architecture
    model.summary()

    # 显示训练集和验证集的acc和loss曲线
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
