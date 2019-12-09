import os
import tensorflow as tf
import numpy as np
from PIL import Image
from imageio import imwrite
import random

def _read_png(file_name):
    im = Image.open(file_name)
    im.thumbnail((256, 256), Image.ANTIALIAS) # resize to (256, 256)
    return np.reshape(np.array(im.getdata())[:,:3], (256, 256, 3))


def get_data(img_names):
    """
    :param img_names: a list of image file full names
    :return: a Tensor of shape [batch size, 256, 256, 3]
    """
    out = [_read_png(img) for img in img_names]
    return tf.convert_to_tensor(out, dtype=tf.float32)/255.0

def get_image_names_labels(class_dir, test_ratio=0.15):
    """
    :param class_dir: a list indicating image dir for class 0, class 1, class 2 ...
    :return: (list of train images, a Tensor of one-hot train labels, list of test images, Tensor of one-hot test labels)
    """
    num_classes = len(class_dir)
    images = []
    labels = []
    for i, dir in enumerate(class_dir):
        imgs = [os.path.join(dir, i) for i in os.listdir(dir) if '.png' in i]
        images.extend(imgs)
        labels.extend([i]*len(imgs))

    tmp = list(zip(images, labels))
    random.shuffle(tmp)
    images, labels = zip(*tmp)
    cut_idx = int(len(images)*test_ratio)

    train_x = list(images[cut_idx:])
    train_y = labels[cut_idx:]
    test_x = list(images[:cut_idx])
    test_y = labels[:cut_idx]

    train_y = tf.one_hot(train_y, num_classes)
    test_y = tf.one_hot(test_y, num_classes)

    return (train_x, train_y, test_x, test_y)



if __name__ == '__main__':
    # x = _read_png('data/ship_detail/d0001.png')
    # imwrite('pic.png', np.reshape(x, (256, 256, 3)))
    tmp = get_image_names_labels(['data/land', 'data/ship_detail'])
    print(tmp)
