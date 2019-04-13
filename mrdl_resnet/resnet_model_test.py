import numpy as np
from resnet50_train import load_model
import cv2

#_1: 0,1,2,3		_2: 0,1,4,5			_3: 2,3,4,5
my_dict_1={'p0':0, 'p1':1, 'p2':2, 'p3':3, 'r0':4, 'r1':5, 'r2':6, 'r3':7, 's0':8, 's1':9, 's2':10, 's3':11}
my_dict_2={'p0':0, 'p1':1, 'p4':2, 'p5':3, 'r0':4, 'r1':5, 'r4':6, 'r5':7, 's0':8, 's1':9, 's4':10, 's5':11}
my_dict_3={'p2':0, 'p3':1, 'p4':2, 'p5':3, 'r2':4, 'r3':5, 'r4':6, 'r5':7, 's2':8, 's3':9, 's4':10, 's5':11}

my_dict_reverse = {v:k for k,v in my_dict_1.items()}

def videotest():
	model = load_model()
	model.load_weights('./model/model_resnet50_D1_1.hdf5')
	cap = cv2.VideoCapture(0)
	while(True):
		ret, frame = cap.read()
		cv2.rectangle(frame, (168, 88), (471, 391), (55,255,155), 2)
		frame1 = frame[90:390, 170:470]
		frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
		#cv2.imwrite('temp.bmp',frame1)
		
		#my_frame = cv2.imread('temp.bmp')
		#img = Image.open('temp.bmp')
		#img =np.array(img)
		frame2 = cv2.GaussianBlur(frame1,(5,5),2)
		frame2 = cv2.adaptiveThreshold(frame2, 2, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
		res, frame2 = cv2.threshold(frame2, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		
		cv2.imwrite('temp.bmp',frame2)
		my_frame = cv2.imread('temp.bmp')
		
		frame3 = np.array(my_frame, dtype = 'f')
		frame3 = frame3/255.0
		frame3 = frame3.reshape([-1,300,300,1])
		pdt = model.predict(frame3)
		result = my_dict_reverse[np.argmax(pdt)]
		frame = cv2.putText(frame, result, (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (55,255,155), 2)

		cv2.imshow('rawinput', frame)
		cv2.imshow('f1', frame2)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

videotest()
