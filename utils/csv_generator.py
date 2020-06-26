'''
以下代码实现功能

将相同的人脸放在同一目录下，不同的人脸放在不同目录下
通过以下代码，可以生成样本对+label
样例如下：
/root/tf-2.0/Siamese_Face_Detection/data/FaceV5_160/458/458_0000.jpg,/root/tf-2.0/Siamese_Face_Detection/data/FaceV5_160/458/458_0001.jpg,1
'''


import os
import csv
import random

# 图片所在的路径
path = '/root/tf-2.0/Siamese_Face_Detection/data/FaceV5_160/'
data_csv_path='/root/tf-2.0/Siamese_Face_Detection/data/FaceV5.csv'
# files列表保存所有类别的路径
files = []
same_pairs = []
different_pairs = []
for file in os.listdir(path):
    if file[0] == '.':
        continue
    file_path = os.path.join(path, file)
    files.append(file_path)
# 该地址为csv要保存到的路径，a表示追加写入


with open(data_csv_path, 'a') as f:
    # 保存相同对
    writer = csv.writer(f)

    for file in files:
        imgs = os.listdir(file)
        imgs = os.listdir(file)
        random.shuffle(imgs)
        if len(imgs)>30:
            imgs=imgs[:30]
        else:
            imgs=imgs
        for i in range(0, len(imgs) - 1):
            for j in range(i + 1, len(imgs)):
                pairs = []
                name = file.split(sep='/')[-1]
                pairs.append(path + name + '/' + imgs[i])
                pairs.append(path + name + '/' + imgs[j])
                pairs.append(1)
                writer.writerow(pairs)


    # 保存不同对
    for i in range(0, len(files) - 1):
        if i+6>len(files):
            limt=len(files)
        else:
            limt=i+6
        for j in range(i + 1, limt):
            filea = files[i]
            fileb = files[j]
            imga_li = os.listdir(filea)
            imgb_li = os.listdir(fileb)
            random.shuffle(imga_li)
            random.shuffle(imgb_li)
            if len(imga_li)>10:
                a_li = imga_li[:10]
            else:
                a_li = imga_li[:]
            if len(imgb_li)>10:
                b_li = imgb_li[:10]
            else:
                b_li = imgb_li[:]
            for p in range(len(a_li)):
                for q in range(len(b_li)):
                    pairs = []
                    name1 = filea.split(sep='/')[-1]
                    name2 = fileb.split(sep='/')[-1]
                    pairs.append(path + name1 + '/' + a_li[p])
                    pairs.append(path + name2 + '/' + b_li[q])
                    pairs.append(0)
                    writer.writerow(pairs)
