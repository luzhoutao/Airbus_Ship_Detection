from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data, read_encodings
from test_utils import IoU, F2 #todo: integrate this to test()

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
		#todo: PLEASE CHECK THIS FUNCTION! when I test, the loss is not decreasing, please help debug!
		#todo: should we use this loss function?  
		labels = tf.dtypes.cast(labels, tf.float32)
		logits = tf.dtypes.cast(logits, tf.float32)
		# loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits) 
		loss = tf.keras.losses.binary_crossentropy(labels, logits, from_logits=True)
		ave_loss = tf.reduce_mean(loss) 
		return ave_loss


	def accuracy(self, logits, labels):
		#logits : numpy of shape=(num_examples, 768,768, 1), dtype=float32
		
		lo1 = tf.reshape(logits, [-1,768,768])
		la1 = tf.reshape(labels, [-1,768,768])
		
		iou = IoU(lo1, la1, eps=1e-6)
		print("-> IoU = %3.8f" %iou)
		

		logits = np.reshape(logits, -1)
		labels = np.reshape(labels, -1)

		#below should work, but caused weird errors: "assignment destination is read-only"?? haven't figure out why, so rewrite using a slower way
		# logits[logits>0.5] = 1 #meaning mask(i.e.,ship)   
		# logits[logits<=0.5] = 0 #meaning background
		# logits = logits.astype(int) #convert to int to compare with labels
		

		#below is slow, should optimize in the future
		logit_copy = []
		for logit in logits:
			if (logit > 0.5):
				logit_copy.append(1)
			else:
				logit_copy.append(0)
		logit_copy = np.reshape(logit_copy, -1)

		accuracy = np.mean(logit_copy == labels) 
		return accuracy

		

def train(model, img_dir, train_img_names,img_to_encodings):
    num_inputs = len(train_img_names)
    steps = int(num_inputs/model.batch_size)

    random.shuffle(train_img_names)

    for i in range(0, steps):
        start = i *model.batch_size
        end = (i+1)*model.batch_size
        # now we load the actual content of the images, which is a huge amount of data
		inputs, labels = get_data(img_dir, train_img_names[start:end],img_to_encodings)


		with tf.GradientTape() as tape:
			logits = model(inputs)
			loss = model.loss(logits, labels)
			
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
				
		if (i % 5 == 0):
			train_acc = model.accuracy(logits, labels)
			print("========>Step %2d, accuracy=%3.4f, loss=%3.4f" %(i, train_acc, loss))



def test(model, img_dir, test_img_names, img_to_encodings):
    num_inputs = len(test_img_names)
    steps = int(num_inputs/model.batch_size)

    accu = []
    for i in range(0, steps):
        start = i *model.batch_size
        end = (i+1)*model.batch_size
        # now we load the actual content of the images, which is a huge amount of data
        inputs, labels = get_data(img_dir, test_img_names[start:end],img_to_encodings)
        logits = model(inputs)
        accuracy = model.accuracy(logits, labels)
        accu.append(accuracy)
    return sum(accu)/len(accu)



def visualize_results(image_inputs):
    #todo
    pass



def main():

    #step1: get the training data and testing data
    img_to_encodings = read_encodings('sample_train.csv')
    img_names = list(img_to_encodings.keys()) # a list of image names as input, images will not be loaded until needed
    train_img_names = img_names[:20]
    test_img_names = img_names[20:30]

    #step2: initialize and train the model
    model = Model(1, 768) #num_class = 1, image_size = 768
    epochs = 5
    for _ in range(epochs):
        train(model, 'sample_jpgs', train_img_names, img_to_encodings)
    #step3: test the model
    accuracy = test(model, 'sample_jpgs', test_img_names, img_to_encodings)
    print("========> Test Accuracy: %.4f" %accuracy)




if __name__ == '__main__':
    main()
