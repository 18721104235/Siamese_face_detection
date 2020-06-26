'''
以下代码实现功能

根据csv_generator生成的样本对，生成训练fit_generator所需的数据集生成器

'''

import os
import cv2
import csv
import tensorflow as tf
import numpy as np
import random
from keras import backend as K
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD,RMSprop
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Concatenate, Add,\
Subtract,GlobalAveragePooling2D,Lambda,AveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam,RMSprop,SGD
from keras.models import load_model

def processImg(filename,h,w):
    """
    :param filename: 图像的路径
    :return: 返回的是归一化矩阵
    """
    height=h
    width=w
    img = cv2.imread(filename)
    img = cv2.resize(img, (height, width))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_to_array(img)
    img /= 255.
    return img

def generator(imgs, batch_size,input_shape):
    """
    自定义迭代器
    :param imgs: 列表，每个包含一对矩阵以及label
    :param batch_size:
    :return:
    """
    h,w,c=input_shape
    while 1:
        random.shuffle(imgs)
        li = imgs[:batch_size]
        pairs = []
        labels = []
        for i in li:
            img1 = i[0]
            img2 = i[1]
            im1 = cv2.imread(img1)
            im2 = cv2.imread(img2)
            if im1 is None or im2 is None:
                continue
            label = int(i[2])
            img1 = processImg(img1,h=h,w=w)
            img2 = processImg(img2,h=h,w=w)
            pairs.append([img1, img2])
            labels.append(label)
        pairs = np.array(pairs)
        labels = np.array(labels)
        yield [pairs[:, 0], pairs[:, 1]], labels


