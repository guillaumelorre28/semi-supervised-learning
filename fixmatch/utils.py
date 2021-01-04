import tensorflow as tf


def sharpen(p, T):

    p = tf.math.pow(p, 1.0/T)
    p = p / tf.reduce_mean(p, axis=-1, keepdims=True)

    return p


def sharpen_uda(p, tau):

    return tf.math.softmax(p/tau)


def mixup(x1, x2, y1, y2, lambda_dist):

    lamb = lambda_dist.sample([tf.shape(x1)[0], 1])
    lamb = tf.maximum(lamb, 1.0-lamb)

    return lamb * x1 + (1 - lamb) * x2, lamb * y1 + (1 - lamb) * y2



