from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
from keras.utils import to_categorical
import numpy as np

from matplotlib import pyplot as plt


def getModel():
    model = Sequential()
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(227, 227, 3), kernel_initializer='uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    #model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    
    model.summary()
    return model
    '''
    
    model = Sequential()
    #第一层卷积网络，使用96个卷积核，大小为11x11步长为4， 要求输入的图片为227x227， 3个通道，不加边，激活函数使用relu
    model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=(227, 227, 3), padding='same', activation='relu',
                     kernel_initializer='uniform'))
    # 池化层
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    # 第二层加边使用256个5x5的卷积核，加边，激活函数为relu
    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    #使用池化层，步长为2
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    # 第三层卷积，大小为3x3的卷积核使用384个
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    
    
    # 第四层卷积,同第三层
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    
    # 第五层卷积使用的卷积核为256个，其他同上
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    
    # model.add(Conv2D(96, (2, 2), strides=(1, 1), padding='same', activation='relu',
    #                kernel_initializer='uniform'))
    # 池化层
    #model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(3, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    
    return model
    '''

'''
def getData(): 
    # 返回图片数据
    train_label = [item.replace('_500', '') for item in os.listdir('./dataset/train/')]
    test_label = [item.replace('_100', '') for item in os.listdir('./dataset/validation/')]
    train_label_with_one_hot = to_categorical(list(range(len(train_label))))
    test_label_with_one_hot = to_categorical(list(range(len(test_label))))
    
    train_data = []
    train_labels = []
    
    test_data = []
    test_labels = []
    
    label = []
    
    # print(os.walk())
    for item in train_label:
        path = './dataset/train/' + item + '_500/'
        for root, _, img_path in os.walk(path):
            for image in img_path:    
                # 读取灰度图
                img = cv2.imread(path + image)
                img = cv2.resize(img, (227, 227), cv2.INTER_AREA)
                train_data.append(img)
                train_labels.append(train_label_with_one_hot[train_label.index(item)])

    
    for item in test_label:
        path = './dataset/validation/' + item + '_100/'
        for root, _, img_path in os.walk(path):
            for image in img_path:    
                # 读取灰度图
                img = cv2.imread(path + image)
                img = cv2.resize(img, (227, 227), cv2.INTER_AREA)
                test_data.append(img)
                test_labels.append(test_label_with_one_hot[test_label.index(item)])

    
    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)


def ImageProcessing(train_data, test_data):
    #将图片进行预处理
    # 图像灰度直方图均衡化处理
    for index, item in enumerate(train_data):
        train_data[index] = cv2.equalizeHist(item)
        
    for index, item in enumerate(test_data):
        test_data[index] = cv2.equalizeHist(item)
        
    # 读图像进行归一化
    train_data = np.expand_dims(train_data / 255, axis=3)
    test_data = np.expand_dims(test_data / 255, axis=3)
    
        
    return train_data, test_data

      

train_data, train_label, test_data, test_label = getData()
# train_data, test_data = ImageProcessing(train_data, test_data)



model = getModel()

# 模型编译
model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['acc'],
    )


history = model.fit(train_data, train_label, epochs=20, batch_size=50)
model.save("./model/model_11_15_17_37.h5")
'''

model = getModel()
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_dir = './dataset/train'
validation_dir = './dataset/validation'

train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(227, 227),
            batch_size=50,
            class_mode='categorical'
        )

validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(227, 227),
            batch_size=10,
            class_mode='categorical'
        )


history = model.fit_generator(
            train_generator,
            steps_per_epoch=28,
            epochs=20,
            validation_data=validation_generator,
            validation_steps=10
        )

model.save('./model_11_16_10_03.h5')
