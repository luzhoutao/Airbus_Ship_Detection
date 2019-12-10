
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from tensorflow.keras.activations import relu

class Classifier(tf.keras.Model):
    def __init__(self, learning_rate = 0.001, num_classes=2):
        super(Classifier, self).__init__()

        self.conv1 = Conv2D(16, (13, 13), input_shape=(256, 256, 3)) # out: (None, 244, 244, 16)
        self.batch_norm1 = BatchNormalization()
        self.pool1 = MaxPooling2D((4, 4)) #(None, 61, 61, 16)

        self.conv2 = Conv2D(32, (6, 6)) #(None, 56, 56, 32)
        self.batch_norm2 = BatchNormalization()
        self.pool2 = MaxPooling2D((2, 2)) #(None, 28, 28, 32)

        self.conv3 = Conv2D(64, (5, 5))#(None, 24, 24, 64)
        self.batch_norm3 = BatchNormalization()
        self.pool3 = MaxPooling2D((4, 4))#(None, 6, 6, 64)

        self.flatten = Flatten()
        self.dense1 = Dense(16, activation='relu')
        self.dense2 = Dense(num_classes, activation='softmax')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


    def call(self, inputs):
        """
        :param inputs: [batch_size, 256, 256, 3]
        :return: probabilities of shape [batch_size, num_classes]
        """
        output1 = self.pool1(relu(self.batch_norm1(self.conv1(inputs))))
        output2 = self.pool2(relu(self.batch_norm2(self.conv2(output1))))
        output3 = self.pool3(relu(self.batch_norm3(self.conv3(output2))))

        output3 = self.flatten(output3)
        return self.dense2(self.dense1(output3))

    def loss(self, probs, labels):
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, probs))

    def accuracy(self, probs, labels):
        """
        :param probs: [batch_size, num_classes]
        :param labels: [batch_size, num_classes]
        :return:
        """
        correct_predictions = tf.equal(tf.argmax(probs, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # def binary_accuracy(self, probs, labels):
    #     # only check whether a ship appears, so separate classes into 2 group: [0~2] vs [3~6]
    #     """
    #     :param probs: [batch_size, 2]
    #     :param labels: [batch_size, 2]
    #     :return:
    #     """
    #     correct_predictions = tf.equal(tf.argmax(probs, 1)>2, tf.argmax(labels, 1)>2)
    #     return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
