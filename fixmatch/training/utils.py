import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.python.ops import math_ops


def compute_class_weights(dataset):
    train_labels = []
    for ex in dataset:
        train_labels.append(ex['label'].numpy())

    labels = np.unique(train_labels)
    class_weight = compute_class_weight('balanced', labels, train_labels)
    class_weight = {k: v for k, v in enumerate(class_weight)}

    return class_weight


class L2PretrainedRegualizer(tf.keras.regularizers.Regularizer):
    
    def __init__(self, l2=1e-4):
        
        self.l2 = l2
        
    def __call__(self, x, y):
        
         return self.l2 * math_ops.reduce_sum(math_ops.square(x-y))
