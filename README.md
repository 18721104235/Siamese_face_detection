face_detection 人脸识别


运行环境：
1. Ubuntu Version: 20.04
2. gcc version: 7.5.0
NVIDIA-SMI: 440.64
Driver Version: 440.64
CUDA Version: 10.2
Cudnn Version: 7.6
GPU: GeForce RTX 2070
Python: 3.7
tensorflow-gpu: 2.0.0
keras: 2.3.1

目标：
通过 Siamese, One Shot learning, Transfer Learning 这里是列表文本三种方式来进行人脸识别三种方式来进行人脸识别
通过accuracy,precision,recall,F1_score检测模型训练情况

目录简介：
data: 用于存放数据集和标签 h5: 用于存放模型
models: 用于存放自定义单支模型，迁移模型权重
utils: 图片裁减代码，样本对csv生成代码，训练集生成器
train*.ipynb: 使用三种方式进行模型训练
predict*.ipynb: 使用不同标准预测结果

