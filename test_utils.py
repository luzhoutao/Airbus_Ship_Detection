import tensorflow as tf
import numpy as np

def IoU(probs, labels, eps=1e-6):
    '''
    Compute IoU for each input in batch. 
    
    [tf.keras.metrics.MeanIoU computes mean IoU over classes may not be proper in our case.]
    :param probs: float tensor, predicted segmentation probabilities [batch_size x height x width].
    :param labels: integer tensor, segmentation mask labels [batch_size x height x width]
    :return: IoU of each input image as a tensor [batch_size]
    '''
    
    # prediction = tf.cast(probs > 0.5, dtype=tf.dtypes.float32) 
    # --(what are we trying to do here? I thought we should convert >0.5 to 1, but here this above line will convert all values to 0.0)

    # intersection = tf.reduce_sum(labels*predictions, axis=[1, 2])  --##stuck here , error: Could not find valid device for node.Node:{{node Mul}}
    # union = tf.reduce_sum(prediction + labels, axis=[1, 2]) - intersection
    # return (intersection + eps) / (union + eps)
    
    prediction = probs.numpy().copy()
    prediction[prediction>0.5] = 1
    prediction[prediction<=0.5] = 0
    labels = np.reshape(labels.numpy(),-1)
    prediction = np.reshape(prediction,-1)
    
    intersection = np.sum(np.multiply(labels,prediction)).mean()
    union = np.sum(prediction + labels).mean()
    return (intersection + eps) / (union + eps)


    


def F2(probs,
       labels,
       thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
       eps=1e-6):
    '''
    Copmute F2 score over a range of IoU threshold.
    
    Note: it is a bit different from what Kaggle was doing. This method treats each input image as one instance rather 
    than each ship in one image. Because we may now only think about the case that at most one ship in an image.
    :param probs: float tensor, predicted segmentation probabilities [batch_size x height x width x 2].
    :param labels: integer tensor, segmentation mask labels [batch_size x height x width]
    :param thresholds: list of IoU thresholds that determine if our predicted mask detects the ship.
    :return: 
    '''
    # Whether there is a ship present in the ground truth.
    present = tf.reduce_sum(labels, axis=[1, 2]) > 0

    scores = []
    for threshold in thresholds:
        # Whether the predicted mask indicates there is a ship.
        positive = tf.reduce_sum(tf.cast(probs > 0.5, dtype=tf.dtypes.float32),
                                 axis=[1, 2]) > 0
        # Whether the predicted mask detects the ship.
        detected = IoU(probs, labels) > threshold

        TP = tf.reduce_sum(
            tf.cast(tf.math.logical_and(present, detected), tf.dtypes.float32))
        FN = tf.reduce_sum(
            tf.cast(tf.math.logical_and(present, tf.math.logical_not(detected)),
                    tf.dtypes.float32))
        FP = tf.reduce_sum(
            tf.cast(tf.math.logical_and(tf.logical_not(present), positive),
                    tf.dtypes.float32))
        scores.append((5 * TP + eps) / (5 * TP + 4 * FN + FP + eps))
    return tf.reduce_mean(scores)


def main():
    # Test IoU: both label and prediction are not empty.
    probs = tf.constant([[[0.1, 0.8], [0.7, 0.1]]])  # 1 x 2 x 2
    labels = tf.constant([[[0., 1.], [1., 1.]]])  # 1 x 2 x 2
    assert (abs(IoU(probs, labels).numpy()[0] - 0.66666) < 1e-5)
    assert (abs(F2(probs, labels) - 0.4) < 1e-5)

    # Test IoU: only label is empty.
    probs = tf.constant([[[0.1, 0.8], [0.7, 0.1]]])  # 1 x 2 x 2 x 2
    labels = tf.constant([[[0., 0.], [0., 0.]]])  # 1 x 2 x 2
    assert (abs(IoU(probs, labels).numpy()[0]) < 1e-5)
    assert (abs(F2(probs, labels)) < 1e-5)

    # Test IoU: both label and prediction are empty.
    probs = tf.constant([[[0.1, 0.2], [0.3, 0.1]]])  # 1 x 2 x 2 x 2
    labels = tf.constant([[[0., 0.], [0., 0.]]])  # 1 x 2 x 2
    assert (abs(IoU(probs, labels).numpy()[0] - 1) < 1e-5)
    assert (abs(F2(probs, labels) - 1) < 1e-5)

    print("All tests pass!")


if __name__ == '__main__':
    main()