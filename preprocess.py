import pickle
import numpy as np
import tensorflow as tf
import os



# def get_data(file_path, batch_size):
def get_data(train=True): 
	#todo:
	images = np.zeros([3,256,256,3], dtype = float)
	labels = np.zeros([3,256,256,1], dtype = int)
	if (train):
		images[0,1,2,1] = 0.23
		images[0,1,2,2] = 0.45
		images[0,1,3,1] = 0.56
		images[0,1,3,2] = 0.78
		labels[0,1,2,0] = 1
		labels[0,1,3,0] = 1
		
		
		images[1,1,2,1] = 0.89
		images[1,1,2,2] = 0.45
		images[1,1,3,1] = 0.56
		images[1,1,3,2] = 0.78
		labels[1,1,2,0] = 1
		labels[1,1,3,0] = 1


		images[2,4,3,1] = 0.89
		images[2,4,3,2] = 0.45
		images[2,4,3,1] = 0.56
		images[2,3,3,2] = 0.78
		labels[2,4,3,0] = 1
		labels[2,4,2,0] = 1
	else:
		images[0,2,2,1] = 0.33
		images[0,2,2,2] = 0.45
		images[0,2,3,1] = 0.06
		images[0,2,3,2] = 0.08
		labels[0,2,2,0] = 1
		labels[0,2,3,0] = 1
		
		
		images[1,4,2,1] = 0.89
		images[1,4,2,2] = 0.45
		images[1,4,3,1] = 0.56
		images[1,4,3,2] = 0.78
		labels[1,4,2,0] = 1
		labels[1,4,3,0] = 1


		images[2,9,3,1] = 0.89
		images[2,9,3,2] = 0.45
		images[2,9,3,1] = 0.56
		images[2,9,3,2] = 0.78
		labels[2,8,3,0] = 1
		labels[2,9,2,0] = 1

	return (images, labels)
