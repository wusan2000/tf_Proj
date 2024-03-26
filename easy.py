# 模拟简单数据
# 思路：Model definition → Model compilation → Training → Evaluation and Prediction

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential  # 引入两个重要的类，Sequential和Dense
from tensorflow.keras.layers import Dense

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽通知消息和警告消息
os.environ['KERAS_BACKEND'] = 'tensorflow'

x = np.linspace(-2, 6, 200)
# 人为地造一组由 y = 0.5x + 2 加上一些噪声而生成的数据
np.random.shuffle(x)
y = 0.5 * x + 2 + 0.15 * np.random.randn(200, )
# plot the data
plt.scatter(x, y)
plt.show()

# 开始构建模型
model = Sequential()
model.add(Dense(units=1, input_dim=1))  # 构建全连接层，此案例全连接层只有一层，而且输入的节点数和输出的节点数都为1
model.compile(loss='mse', optimizer='sgd')  # 默认优化器 'sgd'表示随机梯度下降


# train the first 160 data
x_train, y_train = x[0:160], y[0:160]  # 前160个作为训练集

# start training
# model.fit(x_train, y_train, epochs=100, batch_size=64)
for step in range(0, 500):
    cost = model.train_on_batch(x_train, y_train)
    if step % 20 == 0:
        print('cost is %f' % cost)

# test on the rest 40 data
x_test, y_test = x[160:], y[160:]  # 后40个作为测试集

# start evaluation
cost_eval = model.evaluate(x_test, y_test, batch_size=40)
print('evaluation lost %f' % cost_eval)

model.summary()

w, b = model.layers[0].get_weights()
print('weight %f , bias %f' % (w, b))

# start prediction
y_prediction = model.predict(x_test)
plt.scatter(x_test, y_test)
plt.plot(x_test, y_prediction)
plt.show()
