##C:\Program Files\NVIDIA Corporation\NVSMI>nvidia-smi -l 5
from __future__ import division
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from keras import backend as K
K.backend()
import tensorflow as tf
from PIL import Image
import os
import sys
import time
import cv2
import six
from resnet_v2 import *

#_1: 0,1,2,3		_2: 0,1,4,5			_3: 2,3,4,5
my_dict_1={'p0':0, 'p1':1, 'p2':2, 'p3':3, 'r0':4, 'r1':5, 'r2':6, 'r3':7, 's0':8, 's1':9, 's2':10, 's3':11}
my_dict_2={'p0':0, 'p1':1, 'p4':2, 'p5':3, 'r0':4, 'r1':5, 'r4':6, 'r5':7, 's0':8, 's1':9, 's4':10, 's5':11}
my_dict_3={'p2':0, 'p3':1, 'p4':2, 'p5':3, 'r2':4, 'r3':5, 'r4':6, 'r5':7, 's2':8, 's3':9, 's4':10, 's5':11}

def initializers(training_sets):
	x_data = []
	y_data = []
	input_fold = list(training_sets.keys())
	for item in input_fold:
		imglist = os.listdir('./train/' + item)
		for val in imglist:
			#img = cv2.imread('./train/' + item +'/'+ val)
			img = Image.open('./train/' + item +'/'+ val)
			img = np.array(img)
			x_data.append(img)
			y_data.append(training_sets[item])
	x_data = np.array(x_data,dtype='f')
	x_data = x_data/255.0
	y_data = np.array(y_data)
	print(x_data.shape)
	print(y_data.shape)
	y_data = to_categorical(y_data, num_classes=12)
	x_data, y_data = shuffle(x_data, y_data, random_state=2)
	x_data = x_data.reshape([-1, 300, 300, 1])
	print(x_data.shape)
	print(y_data.shape)
	return x_data, y_data

def load_model():
	model=ResnetBuilder.build_resnet_50((1, 300, 300), 12)
	model.summary()
	model.compile(optimizer='sgd',loss='categorical_crossentropy')
	print('Model Compiled')
	return model

if __name__ == '__main__':
	x_data, y_data = initializers(my_dict_3)
	model = load_model()
	#model.save('./model_h5/resnet50.h5')
	#plot_model(model,to_file='ResNet50.png')
	history = LossHistory()
	
	print("Training start: " + time.asctime(time.localtime(time.time())))
	hist = model.fit(x_data, y_data, batch_size = 16, epochs = 100, verbose = 1, validation_split = 0.1, callbacks=[history])
	history.loss_plot('epoch')
	
	with open('./train_history/resnet50_history_D3_1.txt','w') as f:
		f.write(str(hist.history))
	model.save_weights('./model/model_resnet50_D3_1.hdf5', overwrite = True)
	print("Training end: " + time.asctime(time.localtime(time.time())))

