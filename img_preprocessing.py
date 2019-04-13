"""
Input:
prth:   D:/mrdl_project/raw/
folder: c0: rock    793 
		c1: paper   738
		c2: scissor 789
total: 2320

resize 500*500 --> 300*300
RGB --> Gray

Output:
path:   D:/mrdl_project/train/
folder: 0: rock    793
		1: paper   738
		2: scissor 789
"""
from PIL import Image
import os
import sys
import cv2

def imgpreprocessing1():
	path = os.getcwd()
	path_catalog = os.listdir('./raw')
	print(path)
	print(path_catalog)
	j=0
	for val in path_catalog:
		imglist = os.listdir(path + '/raw/' + val)
		imglist.sort(key= lambda x:int(x[3:-5]))
		imgnum = len(os.listdir(path + '/raw/' + val))
		print(imgnum)
		#print(imglist)
		i=1
		for item in imglist:
			img = Image.open('./raw/'+val+'/'+item)
			img = img.resize((300,300),Image.ANTIALIAS)
			img = img.convert("L")
			img.save('./train/'+str(j)+'/'+str(i)+'.bmp')
			i+=1
		j+=1
		
def imgpreprocessing2():
	path = os.getcwd()
	path_catalog = os.listdir('./raw')
	print(path)
	print(path_catalog)
	j=0
	for val in path_catalog:
		imglist = os.listdir(path + '/raw/' + val)
		imglist.sort(key= lambda x:int(x[3:-5]))
		imgnum = len(os.listdir(path + '/raw/' + val))
		print(imgnum)
		#print(imglist)
		i=1
		for item in imglist:
			img = cv2.imread('./raw/'+val+'/'+item)
			img1 = cv2.resize(img, (300,300))
			img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		
			img2 = cv2.GaussianBlur(img1,(5,5),2)
			img2 = cv2.adaptiveThreshold(img2, 2, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
			res, img2 = cv2.threshold(img2, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
			
			cv2.imwrite('./train1/'+str(j)+'/'+str(i)+'.bmp',img2)
			i+=1
		j+=1

imgpreprocessing2()
