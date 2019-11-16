import cv2


#加载人脸模型
path = './dataset/test/lth_30'
# os.mkdir(path)
face = cv2.CascadeClassifier('./xml/haarcascade_frontalface_alt.xml')
#打开摄像头
capture = cv2.VideoCapture(0)
#创建窗口
cv2.namedWindow('she xiang tou')
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
