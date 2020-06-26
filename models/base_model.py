import os
import cv2
import csv
import tensorflow as tf
import numpy as np
import random
from keras import backend as K
from keras.preprocessing.image import img_to_array
from keras.models import Model, Sequential,load_model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Concatenate, Add
from keras.layers import Subtract,Lambda,AveragePooling2D,ZeroPadding2D,Dropout,GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam,RMSprop,SGD

def base_model(input_shape):

    Ipt = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(Ipt)
    x=Conv2D(32,(4,4),strides=(3,3),padding='same',activation='relu')(x)
    ##Conv bottleneck layer 1
    x=Conv2D(16,(1,1),strides=(1,1),padding='same',activation='relu')(x)
    x=BatchNormalization(axis=3)(x)
    x=Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu')(x)
    x=BatchNormalization(axis=3)(x)
    x=MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid')(x)
    x=Dropout(0.25)(x)
    ##Conv bottleneck layer 2
    x=Conv2D(32,(1,1),strides=(1,1),padding='same',activation='relu')(x)
    x=BatchNormalization(axis=3)(x)
    x=Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu')(x)
    x=BatchNormalization(axis=3)(x)
    x=MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid')(x)
    x=Dropout(0.25)(x)
    ##Conv bottleneck layer 3
    x=Conv2D(64,(1,1),strides=(1,1),padding='same',activation='relu')(x)
    x=BatchNormalization(axis=3)(x)
    x=Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu')(x)
    x=BatchNormalization(axis=3)(x)
    x=MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid')(x)

    #x=GlobalAveragePooling2D()(x)
    x=AveragePooling2D(pool_size=(7,7))(x)
    x=Flatten()(x)
    x=Dense(128,activation='relu')(x)
    x=Dropout(0.1)(x)
    x=Dense(50,activation='relu')(x)

    return Model(Ipt,x)

def base_model_sigmoid(input_shape):

    Ipt = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(Ipt)
    x=Conv2D(32,(4,4),strides=(3,3),padding='same',activation='relu')(x)
    ##Conv bottleneck layer 1
    x=Conv2D(16,(1,1),strides=(1,1),padding='same',activation='relu')(x)
    x=BatchNormalization(axis=3)(x)
    x=Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu')(x)
    x=BatchNormalization(axis=3)(x)
    x=MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid')(x)
    x=Dropout(0.25)(x)
    ##Conv bottleneck layer 2
    x=Conv2D(32,(1,1),strides=(1,1),padding='same',activation='relu')(x)
    x=BatchNormalization(axis=3)(x)
    x=Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu')(x)
    x=BatchNormalization(axis=3)(x)
    x=MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid')(x)
    x=Dropout(0.25)(x)
    ##Conv bottleneck layer 3
    x=Conv2D(64,(1,1),strides=(1,1),padding='same',activation='relu')(x)
    x=BatchNormalization(axis=3)(x)
    x=Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu')(x)
    x=BatchNormalization(axis=3)(x)
    x=MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid')(x)

    #x=GlobalAveragePooling2D()(x)
    x=AveragePooling2D(pool_size=(7,7))(x)
    x=Flatten()(x)
    x=Dense(128,activation='relu')(x)

    return Model(Ipt,x)