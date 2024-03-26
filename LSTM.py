import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, LSTM
from matplotlib import pyplot as plt

input_shape = (17, 2)
num_classes = 2


# Define a simple sequential model
def create_model():
    model = tf.keras.Sequential([
        LSTM(300, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'], )
    return model


# 读取训练数据
def read_data(load_path='', size=0, data_tag=True):
    path1 = load_path + '/0.npy'
    data = np.load(path1)
    if not os.path.exists(path1):
        return
    data = np.reshape(data, (1, input_shape[0], input_shape[1]))
    # size = 62
    read_data_i = 0
    for read_data_i in range(1, size + 1):
        path2 = load_path + '/' + str(read_data_i) + '.npy'
        # arr = arr + np.load(path)
        read_data_tem = []
        try:
            read_data_tem = np.load(path2)
        except Exception as e:
            print('the SIZE in ' + load_path + ' is out of range!', e.args)
        read_data_tem = np.reshape(read_data_tem, (1, 17, 2))
        data = np.append(data, read_data_tem, axis=0)

    # print(data.shape)
    if data_tag:
        train_tag = np.ones((read_data_i + 1, 1))  # 倒地
    else:
        train_tag = np.zeros((read_data_i + 1, 1))  # 站立
    return data, train_tag


def b_test():
    # train
    path1 = './npy(dd)'
    data1, tag1 = read_data(path1, 62, data_tag=True)
    path2 = './npy(zl)'
    data2, tag2 = read_data(path2, 198, data_tag=False)
    train_data = np.append(data1, data2, axis=0)
    train_y = np.append(tag1, tag2, axis=0)

    # test
    path1 = './testdd'
    data1, tag1 = read_data(path1, 14, data_tag=True)
    path2 = './zl'
    data2, tag2 = read_data(path2, 89, data_tag=False)
    test_dd_x = np.append(data1, data2, axis=0)
    test_dd_y = np.append(tag1, tag2, axis=0)
    return train_data, train_y, test_dd_x, test_dd_y


def main():
    train_data, train_y, test_dd_x, test_dd_y = b_test()

    # 打乱数据
    # seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
    np.random.seed(116)  # 使用相同的seed，保证输入特征和标签一一对应
    np.random.shuffle(train_data)
    np.random.seed(116)
    np.random.shuffle(train_y)
    tf.random.set_seed(116)

    print(train_data.shape, train_y.shape, test_dd_x.shape, test_dd_y.shape)
    # Create a basic model instance
    model = create_model()

    checkpoint_save_path = "./checkpoint/lstm.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)
    print(model.trainable_variables)
    file = open('./weights.txt', 'w')
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')
    file.close()

    # model.fit(train_data, train_y, epochs=10)
    history = model.fit(train_data, train_y, batch_size=32, epochs=10,
                        validation_data=(test_dd_x, test_dd_y),
                        validation_freq=1,
                        callbacks=[cp_callback])

    # Display the model's architecture
    model.summary()

    model_path = './model/'
    model.save(model_path)
    score = model.evaluate(test_dd_x, test_dd_y, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

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
