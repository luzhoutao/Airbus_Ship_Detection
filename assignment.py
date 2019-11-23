from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random
# import cv2


class Model(tf.keras.Model):
	def __init__(self, num_class, image_size):
		"""
    	This model class will contain the architecture for your model.
		"""
		super(Model, self).__init__()

		self.num_class = num_class #? 
		self.image_size = image_size
		self.batch_size = 1

		self.dropout_rate = 0.5
		self.learning_rate = 1e-4
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

	def call(self, inputs):
		"""
		Runs a forward pass on an input batch of images.
		:param inputs: images, shape of (num_inputs, ?, ?, ?);
		during training, the shape is (batch_size, ?, ?, ?)
		:return: logits - a matrix of shape (num_inputs, ?, ?, ?); during training, it would be (batch_size, ?, ?, ?)
		"""

		# conv1
		conv1 = tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same')(inputs)
		conv1 = tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same')(conv1)
		pool1   = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')(conv1) #1/2

        # conv2
		conv2 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(pool1)
		conv2 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(conv2)
		pool2   = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')(conv2) # 1/4
        
		# conv3
		conv3 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(pool2)
		conv3 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv3)
		pool3   = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')(conv3) # 1/8

        # conv4
		conv4 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(pool3)
		conv4 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv4)
		conv4 = tf.keras.layers.Dropout(self.dropout_rate)(conv4)
		pool4 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')(conv4)#1/16

		# conv5
		conv5 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool4)
		conv5 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv5)
		conv5 = tf.keras.layers.Dropout(self.dropout_rate)(conv5)
		
		# conv6
		up6   = tf.keras.layers.UpSampling2D(size = (2,2))(conv5)
		concat6 = tf.keras.layers.concatenate([conv4, up6])
		conv6 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(concat6)
		conv6 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv6)
        
		# conv7
		up7 = tf.keras.layers.UpSampling2D(size = (2,2))(conv6)
		# up7 = tf.keras.layers.Conv2D(32, 2, activation='relu', padding='same')(up7)
		concat7 = tf.keras.layers.concatenate([conv3, up7],axis = 3) 
		conv7 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(concat7)
		conv7 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv7)

		# conv8
		up8 = tf.keras.layers.UpSampling2D(size = (2,2))(conv7)
		# up8 = tf.keras.layers.Conv2D(16, 2, activation='relu', padding='same')(up8)
		concat8 = tf.keras.layers.concatenate([conv2,up8],axis = 3)
		conv8 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(concat8)
		conv8 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(conv8)

		# conv9
		up9 = tf.keras.layers.UpSampling2D(size = (2,2))(conv8)
		# up9 = tf.keras.layers.Conv2D(8, 2, activation='relu', padding='same')(up9) #todo
		concat9 = tf.keras.layers.concatenate([conv1,up9], axis = 3)
		conv9 = tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same')(concat9)
		conv9 = tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same')(conv9)

		# drop9   = tf.keras.layers.Dropout(self.dropout_rate)(conv9) #add an additional dropout here 
		# conv9 = tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same')(conv9)
		# conv10
		logits = tf.keras.layers.Conv2D(self.num_class, 1, activation='sigmoid', padding='same')(conv9) #kernal_size = 1
		return logits
		

	def loss(self, logits, labels):
		#todo: should we use this loss function? 
		labels = tf.dtypes.cast(labels, tf.float32)
		logits = tf.dtypes.cast(logits, tf.float32)
		loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits)
		ave_loss = tf.reduce_mean(loss) 
		return ave_loss


	def accuracy(self, logits, labels):
		logits = logits.numpy() #convert tensor to numpy for following processing
		# labels = labels.numpy() #labels are already numpy	
		logits[logits>0.5] = 1 #meaning mask(i.e.,ship)
		logits[logits<=0.5] = 0 #meaning background
		logits = logits.astype(int) #convert to int to compare with labels
		
		accuracy = np.mean(logits == labels)
		return accuracy
		

def train(model, train_inputs, train_labels):

	train_inputs = tf.image.random_flip_left_right(train_inputs)

	(num_inputs, _,_,_) = train_inputs.shape
	indices = tf.range(num_inputs)
	indices = tf.random.shuffle(indices)
	train_inputs = tf.gather(train_inputs, indices)
	train_labels = tf.gather(train_labels, indices)

	steps = int(num_inputs/model.batch_size)
	
	for i in range(0, steps):
		start = i *model.batch_size
		end = (i+1)*model.batch_size
		inputs = train_inputs[start:end,:,:,:]
		labels = train_labels[start:end, :, :, :]

		with tf.GradientTape() as tape:
			logits = model(inputs)
			loss = model.loss(logits, labels)
			train_acc = model.accuracy(logits, labels)
			print("===========>Accuracy on training set after {} training steps: {}".format(i, train_acc))
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))



def test(model, test_inputs,test_labels):
	logits = model(test_inputs)
	accuracy = model.accuracy(logits, test_labels)
	return accuracy

def visualize_results(image_inputs):
	#todo
	pass



def main():
	#step1: get the training data and testing data
	(train_inputs, train_labels) = get_data() #todo: change 
	(test_inputs, test_labels) = get_data(train=False)#todo: change 

	#step2: initialize and train the model
	model = Model(1, 256) #num_class = 1, image_size = 256
	epochs = 10
	for _ in range(epochs): 
		train(model, train_inputs, train_labels)

	#step3: test the model
	accuracy = test(model, test_inputs, test_labels)
	print("========> Test Accuracy: %.4f" %accuracy)



if __name__ == '__main__':
	main()