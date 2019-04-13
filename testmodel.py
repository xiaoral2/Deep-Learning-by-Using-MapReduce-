##C:\Program Files\NVIDIA Corporation\NVSMI>nvidia-smi -l 5
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import random
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils.np_utils import to_categorical
from keras import backend as K
K.backend()
import tensorflow as tf
import os
import sys
import cv2
from gesrecog import loadCNN
from PIL import Image

gesdict={0:'rock', 1:'paper', 2:'scissor'}

def imgtest():
	model = loadCNN()
	model.load_weights('./model_2.hdf5')
	L=[]
	for i in range(1,793):
		testimg = Image.open('./train1/0/'+str(i)+'.bmp')
		testimg = np.array(testimg, dtype = 'f')
		testimg = testimg/255.0
		testimg = testimg.reshape([-1,300,300,1])
		pdt = model.predict(testimg)
		L.append(np.argmax(pdt))
		#print(gesdict[np.argmax(pdt)])
	print(L)
	print(L.count(0))

def videotest():
	model = loadCNN()
	model.load_weights('./model_2.hdf5')
	cap = cv2.VideoCapture(0)
	while(True):
		ret, frame = cap.read()
		frame1 = cv2.resize(frame, (300,300))
		frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
		
		frame2 = cv2.GaussianBlur(frame1,(5,5),2)
		frame2 = cv2.adaptiveThreshold(frame2, 2, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
		res, frame2 = cv2.threshold(frame2, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		
		frame3 = np.array(frame2, dtype = 'f')
		frame3 = frame3/255.0
		frame3 = frame3.reshape([-1,300,300,1])
		pdt = model.predict(frame3)
		result = gesdict[np.argmax(pdt)]
		frame = cv2.putText(frame, result, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (55,255,155), 2)
		cv2.imshow('rawinput', frame)
		cv2.imshow('frame2', frame2)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

def videotest2():
	model = loadCNN()
	model.load_weights('./model_2.hdf5')
	cap = cv2.VideoCapture(0)
	while(True):
		ret, frame = cap.read()
		cv2.rectangle(frame, (168, 88), (471, 391), (55,255,155), 2)
		frame1 = frame[90:390, 170:470]
		frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
		
		frame2 = cv2.GaussianBlur(frame1,(5,5),2)
		frame2 = cv2.adaptiveThreshold(frame2, 2, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
		res, frame2 = cv2.threshold(frame2, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		
		frame3 = np.array(frame2, dtype = 'f')
		frame3 = frame3/255.0
		frame3 = frame3.reshape([-1,300,300,1])
		pdt = model.predict(frame3)
		result = gesdict[np.argmax(pdt)]
		frame = cv2.putText(frame, result, (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (55,255,155), 2)
		
		cv2.imshow('rawinput', frame)
		cv2.imshow('f1', frame2)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

#imgtest()
videotest2()
