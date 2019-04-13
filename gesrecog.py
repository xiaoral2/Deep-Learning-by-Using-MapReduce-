"""
http://www.cnblogs.com/aoru45/p/9748475.html
https://github.com/asingh33/CNNGestureRecognizer
https://keras.io/backend/#function
"""
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from keras import backend as K
K.backend()
import tensorflow as tf
from PIL import Image
import os
import sys
import time

def loadCNN():
	global get_output
	model = Sequential()
	model.add(Conv2D(32,(5,5), padding="valid", input_shape=(300,300,1)))
	convout1 = Activation("relu")
	model.add(convout1)
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64,(3,3)))
	convout2 = Activation("relu")	
	model.add(convout2)
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64,(5,5)))
	convout3 = Activation("relu")
	model.add(convout3)
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64,(5,5)))
	convout4 = Activation("relu")
	model.add(convout4)
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation("relu"))
	model.add(Dropout(0.5))
	model.add(Dense(128))
	model.add(Activation("relu"))
	model.add(Dropout(0.5))
	model.add(Dense(3))
	model.add(Activation("softmax"))
	model.compile(loss = "categorical_crossentropy", optimizer = "adadelta", metrics = ['accuracy'])
	model.summary()
	config = model.get_config()
	layer = model.layers[11]
	get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
	return model

def initializers():
	x_data = []
	y_data = []
	for i in range(3):
		imglist = os.listdir('./train1/' + str(i))
		for item in imglist:
			img = Image.open('./train1/' + str(i) +'/'+ item)
			img = np.array(img)
			x_data.append(img)
			y_data.append(i)
	x_data = np.array(x_data,dtype='f')
	x_data = x_data/255.0       
	y_data = np.array(y_data)
	#print(x_data.shape)
	#print(y_data.shape)
	y_data = to_categorical(y_data, num_classes=3)
	x_data, y_data = shuffle(x_data, y_data, random_state=2)
	x_data = x_data.reshape([-1, 300, 300, 1])
	print(x_data.shape)
	print(y_data.shape)
	return x_data, y_data
	
if __name__ == '__main__':
	x_data, y_data = initializers()
	model = loadCNN()
	print("Training start: " + time.asctime(time.localtime(time.time())))
	hist = model.fit(x_data, y_data, batch_size = 32, epochs = 5, verbose = 1, validation_split = 0.1)
	model.save_weights('./model/model_3.hdf5', overwrite = True)
	print("Training end: " + time.asctime(time.localtime(time.time())))

