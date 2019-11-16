import cv2
from keras.models import load_model
import numpy as np
import os


#加载人脸模型
face = cv2.CascadeClassifier('./xml/haarcascade_frontalface_alt.xml')
#打开摄像头
capture = cv2.VideoCapture(0)
#创建窗口
cv2.namedWindow('Face Recognizition')

# 标签
label = {}

for index, item in enumerate(os.listdir('./dataset/train/')):
    label[index] = item.replace('_500', '')

# 导入模型
model = load_model('./model/model_11_15_17_37.h5')

font = cv2.FONT_HERSHEY_SIMPLEX
i = 0
while True:      
    ret, frame = capture.read()
    if frame is not None:
        cv2.imshow('gc',frame)
        #图片灰度调整	
        
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        #检查人脸s
        faces = face.detectMultiScale(gray,1.1,3,0,(100,100))
        # new_img = 
        for(x,y,w,h) in faces:
            i = i + 1
            # tmp_img = cv2.resize(frame[y:y+h,x:x+w],(200,200))
            tmp_img = cv2.resize(gray[y+int(0.05*h):y+int(h*0.95), x+int(w*0.05):x+int(w*0.95)],(100,100))
            # cv2.imwrite("./imgafterresize/" + str(i) + ".jpg", tmp_img)
            tmp_img_1 = cv2.equalizeHist(tmp_img)
            # raw = cv2.resize(tmp_img, (100, 100), cv2.INTER_AREA) / 255
            # tmp_img = cv2.resize(tmp_img[y+10: y+h-10, x+10: x+w-10], (100, 100))
            raw = tmp_img_1 / 255 
            data = np.expand_dims(raw, axis=0)
            data = np.expand_dims(data, axis=3)
            result = model.predict(data)
            name = label[np.argmax(result)] + ': ' + str(result[0][np.argmax(result)])
            print(name)
            cv2.rectangle(frame,(int(x+int(0.05*w)), int(y+int(0.05*h))),(int(x+int(w*0.95)), int(y+int(h*0.95))),(0, 0, 255),2)
            # cv2.putText(frame, '000', (50, 100), font, (0, 0, 255))
            if result[0][np.argmax(result)] >= 0.90:
                cv2.putText(frame, name, (x + 30, y - 8), font, 1., (255, 0, 0), 3)
            else:
                cv2.putText(frame, "Stranger", (x + 30, y - 8), font, 1., (255, 0, 0), 3)
            cv2.imshow('gc',frame)
            # cv2.putText(frame, 'liuzufeng', (10, 10), (0, 0, 255), 12)
            
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            '''
            # img = Image.open(frame)
            print(frame)
            raw = np.array(img) / 255
            data = np.expand_dims(raw, axis=0)
            result = model.predict(data)
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, label[np.argmax(result)], ((x + w/2 - 6), (y + 10)), (0, 0, 255), 12)
            cv2.imshow('gc',frame)
            '''
        if(cv2.waitKey(5) & 0xFF == ord('q')):
            break
        


capture.release()
cv2.destroyAllWindows() 
         











'''
#获取实时画面
count=40
num=0
while True:
	
	if num == 30:
	 	break
	 #读取帧画面
	ret,frame = capture.read()
	#图片灰度调整	
	gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
	#检查人脸
	faces = face.detectMultiScale(gray,1.1,3,0,(100,100))
	print(type(frame))
	print(frame.shape)
	print(frame.ndim)
	print(type(faces))

	print("-----------")
	 #标记人脸
	for(x,y,w,h) in faces:
	 	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
	 	#显示
	 	new_frame=cv2.resize(frame[y:y+h,x:x+w],(200,200))
	 	cv2.imwrite('%s/%s.png'%(path,str(count)),new_frame)

	 	count += 1
	 	num+=1
	 	cv2.imshow('gc',frame)
	 	if (cv2.waitKey(5) & 0xFF == ord('q')):
	 		break

capture.release()
cv2.destroyAllWindows()

# img = cv2.imread('timg.jpg')
# #加载人脸模型
# face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# #调整灰度
# gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# #检查人脸
# faces = face.detectMultiScale(gray)
'''