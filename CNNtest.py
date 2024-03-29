#This program is a practice for EEG signal classification in CNN.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D, Dropout, Conv2D, Activation, MaxPool2D, \
    GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import layers
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras.preprocessing.image import *
import h5py
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
import os
from sklearn.model_selection import train_test_split

# 训练样本目录和验证样本目录
train_dir = '/ '  # 训练集20000条
validation_dir = '/ '  # 验证集5000条
# 对训练图像进行数据增强
train_datagen = ImageDataGenerator(rescale=1. / 255,  # 数据归一化
                                   rotation_range=30,  # 图片随机旋转的最大角度
                                   width_shift_range=0.2,  # 图片在水平位置上平移的最大百分比值
                                   height_shift_range=0.2,  # 图片在竖直位置上平移的最大百分比值
                                   shear_range=0.2,  # 随机错位切换的角度
                                   zoom_range=0.2,  # 图片随机缩放的范围
                                   horizontal_flip=True)  # 随机将一半的图片进行水平翻转
# 对验证图像只将图片归一化，不进行增强变换
validation_datagen = ImageDataGenerator(rescale=1. / 255)
# 利用flow_from_directory 函数生成训练数据
training_set = train_datagen.flow_from_directory(train_dir,  # 训练集文件夹路径
                                                 target_size=(100, 100),  # 每张图片的size
                                                 batch_size=16,  # 每个batch的大小
                                                 class_mode='categorical',  # 标签的类型
                                                 shuffle=True)  # 是否打乱
# 利用flow_from_directory 函数生成验证数据
validation_set = validation_datagen.flow_from_directory(validation_dir,
                                                        target_size=(100, 100),
                                                        batch_size=16,
                                                        class_mode='categorical',
                                                        shuffle=False)

label_test = np.empty([300])        #生成y lable数据，且符合纬度,   30599
label_trian = np.empty([300])
for i in range(0, 3):
    label_trian[100*i:100*i+100] = i
    label_test[100*i:100*i+100] = i

#print(np.shape(training_set), np.shape(validation_set))
print("data loaded")

# CNN
# 初始化模型
model = Sequential()
# 卷积层
model.add(Conv2D(16, (3, 3),  # 16个卷积核，卷积核大小为(3,3)
                 input_shape=(100, 100, 3),  # 输入图像大小
                 activation='relu'))  # 激活函数选取relu
# 池化层
model.add(MaxPool2D(pool_size=(2, 2)))  # 池化大小
# 卷积层
model.add(Conv2D(16, (3, 3), activation='relu'))
# 池化层
model.add(MaxPool2D(pool_size=(2, 2)))
# Flatten
model.add(Flatten())  # 为全连接层做准备，拉成一维数据
# 全连接层
model.add(Dense(units=128, activation='relu'))
# 输出层
model.add(Dense(len(training_set.class_indices), activation='softmax'))
# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])  # 设置损失函数，优化器，模型在训练和测试时的性能指标
# 最后打印网络结构
model.summary()

# 训练模型
model.fit(training_set,  # 训练集
                    validation_data=validation_set,  # 验证集
                    epochs=8)  # 10个epoch
model.save('CNN.h5')  # 保存模型（.h5文件格式）

