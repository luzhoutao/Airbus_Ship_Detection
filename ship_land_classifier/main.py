import tensorflow as tf
from model import Classifier
from preprocess import get_image_names_labels, get_data
import os
from sys import argv

def train(model, train_inputs, train_labels, batch_size=64):

    assert type(train_inputs)==list
    # shuffle randomly
    num_examples = len(train_inputs)
    # shuffled_indices = tf.random.shuffle(list(range(num_examples)))
    # train_inputs = tf.image.random_flip_left_right(tf.gather(train_inputs, shuffled_indices))
    # train_labels = tf.gather(train_labels, shuffled_indices)

    for i in range(0, num_examples - batch_size + 1, batch_size):
        x = get_data(train_inputs[i:i + batch_size])
        y = train_labels[i:i + batch_size]
        with tf.GradientTape() as tape:
            probs = model(x)
            loss = model.loss(probs, y)

            accu = model.accuracy(probs, y)
            # if i % 500 == 0:
            print("%g/%g - accu: %g" %(i, num_examples, accu))
            print('loss: %g'%loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels, batch_size=64):
    accus = []
    num_examples = len(test_inputs)
    for i in range(0, num_examples - batch_size + 1, batch_size):
        x = get_data(test_inputs[i:i + batch_size])
        y = test_labels[i:i + batch_size]
        probs = model(x)
        accus.append(model.accuracy(probs, y))
        print('Tested %g/%g - ave accus: %s' % (i, num_examples, sum(accus)/len(accus)))
    return sum(accus) / len(accus)

def main(learning_rate=0.001, batch_size=64, num_classes=2, num_epoches=1,
         img_dir="./data"):
    land_dir = img_dir+'/land'
    ship_dir = img_dir+'/detail'

    train_x, train_y, test_x, test_y = get_image_names_labels([land_dir, ship_dir], test_ratio=0.3)

    model = Classifier(learning_rate=learning_rate, num_classes=num_classes)

    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    if os.path.exists(checkpoint_dir):
        print("Reading checkpoints")
        checkpoint.restore(manager.latest_checkpoint)


    if len(argv)==1:
        # no extra cli parameter
        for i in range(num_epoches):
            print("Epoch: ", i)
            train(model, train_x, train_y, batch_size)
            print("Saving checkpoints")
            manager.save()
    else:
        print('skip training')

    accu = test(model, test_x, test_y, batch_size)
    print("Test accuracy:", accu)

if __name__ == '__main__':
    main()

