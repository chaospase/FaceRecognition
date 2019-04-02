# 引入包
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


# class GraySale:
def main():
    gray_scale_first('./芝麻.jpg')
    # gray_scale_second()


def gray_scale_first(img_path):
    # 读取文件
    zhima = plt.imread(img_path)
    plt.imshow(zhima)
    plt.show()
    # 转变需要的类型，并且产生噪声，显示最后的图片
    zhima_noisy = zhima.copy().astype(float)
    zhima_noisy += zhima_noisy.std() * 0.3 * np.random.standard_normal(zhima_noisy.shape)
    plt.imshow(zhima_noisy)
    # 查看形状
    print(zhima.shape)
    # 输出结果为：(662, 1000, 3)
    # 平均值法
    # 聚合操作后就减少了一个维度了
    zhima_mean = zhima.mean(axis=2)
    print(zhima_mean.shape)
    # 输出结果为：(662, 1000)
    plt.imshow(zhima_mean, cmap='gray')
    # 最大值法
    zhima_max = zhima.max(axis=-1)
    print(zhima_max.shape)
    plt.imshow(zhima_max, cmap='gray')
    # RGB三原色法
    gravity = np.array([0.299, 0.587, 0.114])
    # red*0.299+green*0.587+blue*0.114
    # 矩阵乘法
    zhima_gravity = np.dot(zhima, gravity)
    zhima_gravity.shape
    plt.imshow(zhima_gravity, cmap='gray')
    plt.show()
    plt.savefig("gray1.png")


def gray_scale_second():
    mnist = read_data_sets('MNIST_data', one_hot=False)
    x, y = mnist.test.next_batch(1)
    x = x.reshape([28, 28])

    fig = plt.figure()
    # Method1
    ax1 = fig.add_subplot(221)
    ax1.imshow(x, cmap=plt.cm.gray)

    # Method2: 反转色
    ax2 = fig.add_subplot(222)
    ax2.imshow(x, cmap=plt.cm.gray_r)  # r表示reverse

    # Method3（等价于Method1）
    ax3 = fig.add_subplot(223)
    ax3.imshow(x, cmap='gray')

    # Method4（等价于Method2）
    ax4 = fig.add_subplot(224)
    ax4.imshow(x, cmap='gray_r')

    plt.show()
    plt.savefig("./gray2.png")


if __name__ == '__main__':
    main()
