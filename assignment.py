from __future__ import absolute_import

import tensorflow as tf

import os
import numpy as np
import random
import argparse
from matplotlib import pyplot as plt

from preprocess import get_data, read_encodings
from test_utils import IoU, F2


gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)

## --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Airbus Ship Detection')

parser.add_argument('--encoding-file', type=str, default='sample_train.csv',
                    help='path to encoding file')

parser.add_argument('--img-dir', type=str, default='sample_jpgs',
                    help='path to image directory')

parser.add_argument('--mode', type=str, default='train',
                    help='Can be "train" or "test"')

parser.add_argument('--restore-checkpoint', action='store_true',
                    help='Use this flag if you want to resuming training from a previously-saved checkpoint')

parser.add_argument('--batch-size', type=int, default=4,
                    help='Sizes of image batches fed through the network')

parser.add_argument('--num-epochs', type=int, default=5,
                    help='Number of passes through the training data to make before stopping')

parser.add_argument('--learn-rate', type=float, default=1e-4,
                    help='Learning rate for Adam optimizer')

parser.add_argument('--dropout-rate', type=float, default=0.5,
                    help='Dropout rate')

parser.add_argument('--log-every', type=int, default=10,
                    help='Print losses after every [this many] training iterations')

parser.add_argument('--save-every', type=int, default=100,
                    help='Save the state of the network after every [this many] training iterations')

parser.add_argument('--device', type=str, default='GPU:0' if gpu_available else 'CPU:0',
                    help='specific the device of computation eg. CPU:0, GPU:0, GPU:1, GPU:2, ... ')

args = parser.parse_args()

## --------------------------------------------------------------------------------------


class Model(tf.keras.Model):
    def __init__(self, num_class, image_size):
        """
        This model class will contain the architecture for your model.
        """
        super(Model, self).__init__()

        self.num_class = num_class
        self.image_size = image_size
        self.batch_size = args.batch_size

        self.dropout_rate = args.dropout_rate
        self.learning_rate = args.learn_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # conv1
        self.conv1_1 = tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same')
        self.conv1_2 = tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')  # 1/2

        # conv2
        self.conv2_1 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')
        self.conv2_2 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')  # 1/4

        # conv3
        self.conv3_1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')
        self.conv3_2 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')
        self.pool3 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')  # 1/8

        # conv4
        self.conv4_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.conv4_2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.pool4 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')  # 1/16

        # conv5
        self.conv5_1 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.conv5_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)

        # conv6
        self.up6 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.concat6 = tf.keras.layers.Concatenate(axis=3)
        self.conv6_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.conv6_2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')

        # conv7
        self.up7 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.concat7 = tf.keras.layers.Concatenate(axis=3)
        self.conv7_1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')
        self.conv7_2 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')

        # conv8
        self.up8 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.concat8 = tf.keras.layers.Concatenate(axis=3)
        self.conv8_1 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')
        self.conv8_2 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')

        # conv9
        self.up9 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.concat9 = tf.keras.layers.Concatenate(axis=3)
        self.conv9_1 = tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same')
        self.conv9_2 = tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same')

        self.out = tf.keras.layers.Conv2D(self.num_class, 1, activation='sigmoid', padding='same')  # kernal_size = 1

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, ?, ?, ?);
        during training, the shape is (batch_size, ?, ?, ?)
        :return: logits - a matrix of shape (num_inputs, ?, ?, ?); during training, it would be (batch_size, ?, ?, ?)
        """

        # conv1
        conv1 = self.conv1_2(self.conv1_1(inputs))
        pool1 = self.pool1(conv1) #1/2

        # conv2
        conv2 = self.conv2_2(self.conv2_1(pool1))
        pool2 = self.pool2(conv2) # 1/4

        # conv3
        conv3 = self.conv3_2(self.conv3_1(pool2))
        pool3 = self.pool3(conv3) # 1/8

        # conv4
        conv4 = self.dropout1(self.conv4_2(self.conv4_1(pool3)))
        pool4 = self.pool4(conv4) # 1/16

        # conv5
        conv5 = self.dropout2(self.conv5_2(self.conv5_1(pool4)))

        # conv6
        up6 = self.up6(conv5)
        concat6 = self.concat6([conv4, up6])
        conv6 = self.conv6_2(self.conv6_1(concat6))

        # conv7
        up7 = self.up7(conv6)
        # up7 = tf.keras.layers.Conv2D(32, 2, activation='relu', padding='same')(up7)
        concat7 = self.concat7([conv3, up7])
        conv7 = self.conv7_2(self.conv7_1(concat7))

        # conv8
        up8 = self.up8(conv7)
        # up8 = tf.keras.layers.Conv2D(16, 2, activation='relu', padding='same')(up8)
        concat8 = self.concat8([conv2, up8])
        conv8 = self.conv8_2(self.conv8_1(concat8))

        # conv9
        up9 = self.up9(conv8)
        # up9 = tf.keras.layers.Conv2D(8, 2, activation='relu', padding='same')(up9) #todo
        concat9 = self.concat9([conv1,up9])
        conv9 = self.conv9_2(self.conv9_1(concat9))

        # drop9   = tf.keras.layers.Dropout(self.dropout_rate)(conv9) #add an additional dropout here 
        # conv9 = tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same')(conv9)
        # conv10
        logits = self.out(conv9) #kernal_size = 1
        return logits
		

    def loss(self, logits, labels):
        labels = tf.dtypes.cast(labels, tf.float32)
        logits = tf.dtypes.cast(logits, tf.float32)
        loss = tf.keras.losses.binary_crossentropy(labels, logits, from_logits=False)
        ave_loss = tf.reduce_mean(loss)
        return ave_loss


    def accuracy(self, logits, labels):
        '''
        Compute per-pixel accuracy and IOU score.
        :param logits: logits returned from model
        :param labels: label for inputs
        :return: 
        '''
        accu = np.mean(tf.cast(logits > 0.5, dtype=tf.dtypes.int32).numpy() == labels)
        iou = tf.reduce_mean(IoU(logits[:, :, :, 0], labels[:, :, :, 0])).numpy()
        return accu, iou

		
def train(model, img_dir, train_img_names, img_to_encodings, manager):
    num_inputs = len(train_img_names)
    steps = int(num_inputs / model.batch_size)

    random.shuffle(train_img_names)

    for i in range(0, steps):
        start = i * model.batch_size
        end = (i + 1) * model.batch_size
        # now we load the actual content of the images, which is a huge amount of data
        inputs, labels = get_data(img_dir, train_img_names[start:end],img_to_encodings)

        with tf.GradientTape() as tape:
            logits = model(inputs)
            loss = model.loss(logits, labels)
			
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
				
        if i % args.log_every == 0:
            train_acc, train_iou = model.accuracy(logits, labels)
            print("========>Step %2d, accuracy = %3.4f, loss = %3.4f, IoU = %3.4f" % (i, train_acc, loss, train_iou))

        if i % args.save_every == 0:
            manager.save()


def test(model, img_dir, test_img_names, img_to_encodings):
    num_inputs = len(test_img_names)
    steps = int(num_inputs / model.batch_size)

    log_accu, log_iou = [], []
    for i in range(0, steps):
        start = i * model.batch_size
        end = (i + 1) * model.batch_size
        # now we load the actual content of the images, which is a huge amount of data
        inputs, labels = get_data(img_dir, test_img_names[start:end],img_to_encodings)
        logits = model(inputs)
        accuracy, iou = model.accuracy(logits, labels)
        log_accu.append(accuracy)
        log_iou.append(iou)
    return np.mean(log_accu), np.mean(log_iou)


def visualize_results(image_inputs):
    #todo
    pass

def balance_sample_dataset(img_to_encodings):
    empty_images, nonempty_images = [], []

    for name, encodings in img_to_encodings.items():
        if len(encodings) == 1 and encodings[0] == "\n":
            empty_images.append(name)
        else:
            nonempty_images.append(name)

    img_names = []
    img_names.extend(random.sample(empty_images, len(nonempty_images)))
    img_names.extend(nonempty_images)

    return img_names

def main():
    VALIDATION_RATE = 0.1

    #step1: get the training data and testing data
    img_to_encodings = read_encodings(args.encoding_file)
    img_names = balance_sample_dataset(img_to_encodings)
    N = len(img_names)

    # step2: initialize and train the model
    model = Model(num_class=1, image_size=768)

    # For saving/loading models
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    if args.mode == "test" or args.restore_checkpoint:
        # restores the latest checkpoint using from the manager
        checkpoint.restore(manager.latest_checkpoint)

    if args.mode == "train":
        train_img_names = img_names[int(N * VALIDATION_RATE):]
        val_img_names = img_names[:int(N * VALIDATION_RATE)]

        for e in range(args.num_epochs):
            print("Epoch %d:" % e)
            train(model, args.img_dir, train_img_names, img_to_encodings, manager)

        #step3: test the model
        accuracy, iou = test(model, args.img_dir, val_img_names, img_to_encodings)
        print("========> Validation: Accuracy = %.4f, IoU = %.4f" % (accuracy, iou))

    if args.mode == "test":
        accuracy, iou = test(model, args.img_dir, val_img_names, img_to_encodings)
        print("========> Test: Accuracy = %.4f, IoU = %.4f" % (accuracy, iou))


if __name__ == '__main__':
    main()
