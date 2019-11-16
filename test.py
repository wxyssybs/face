# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 22:47:40 2019

@author: Zufeng Liu
"""

from keras.models import load_model
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
import cv2


'''
label = {}

for index, item in enumerate(os.listdir('./dataset/train/')):
    label[index] = item.replace('_500', '')
    

model = load_model('./model/model_1.h5')

# img = Image.open('./validation/12.png')
# img = cv2.imread('./dataset/validation/yhn_100/67.png')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

raw = img_gray / 255
print(raw.shape)
data = np.expand_dims(raw, axis=0)
data = np.expand_dims(data, axis=3)
result = model.predict(data)
print(data.shape)

plt.imshow(np.array(img))

print(result)

print(label)
print('识别结果为：', label[np.argmax(result)])
print('准确率为：', format(result[0][np.argmax(result)], '.4f'))
'''



img = cv2.imread('./dataset/train/yhn_500/0.png', 0)
print(img.shape)
new_img = cv2.equalizeHist(img)
cv2.imshow("img", new_img)
cv2.waitKey(5)