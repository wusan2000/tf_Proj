import numpy as np


def ycl(a):
    a1 = a[:, 0]
    a2 = a[:, 1]
    b1 = a1.nonzero()
    c1 = a1[b1].min()
    b2 = a2.nonzero()
    c2 = a2[b2].min()
    max = a2[b2].max()
    z = max - c2
    a1 = a1 - c1
    a2 = a2 - c2
    a1 = a1 / z * 100
    a2 = a2 / z * 100
    c = np.zeros((17, 2))
    c[:, 0] = a1
    c[:, 1] = a2
    c = abs(c)
    return c


a = np.load('./source/DATA_2_origin/2/timber/1/0.npy')
print(a)
b = np.load('./source/DATA_2_0_100/2/timber/1/0.npy')
print(b)

