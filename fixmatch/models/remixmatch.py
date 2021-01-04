import tensorflow as tf
from fixmatch.utils import sharpen, mixup
import tensorflow_probability as tfp

layers = tf.keras.layers


class ReMixmatch(tf.keras.Model):

    def __init__(self, backbone, T, lambda_u, K, alpha, num_classes=10):
        super(ReMixmatch, self).__init__()

        self.backbone = backbone
        self.T = T
        self.lambda_u = lambda_u
        self.num_classes = num_classes
        self.K = K

        self.lambda_dist = tfp.distributions.Beta(alpha, alpha)

        self.sup_loss = tf.keras.losses.CategoricalCrossentropy()

        self.unsup_loss_tracker = tf.keras.metrics.Mean(name="unsup_loss")
        self.sup_loss_tracker = tf.keras.metrics.Mean(name="sup_loss")

        self.unsup_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="unsup_accuracy")
        self.sup_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="sup_accuracy")

    @property
    def metrics(self):
        _metrics = [
            self.unsup_loss_tracker,
            self.sup_loss_tracker,
            self.unsup_accuracy_tracker,
            self.sup_accuracy_tracker
        ]

        return _metrics

    @property
    def trainable_variables(self):
        trainable_vars = (
            self.backbone.trainable_variables
        )

        return trainable_vars

    def train_step(self, inputs):
        inputs_sup, inputs_unsup = inputs
        x, y = inputs_sup
        x_u = inputs_unsup

        s_u = tf.shape(x_u)

        with tf.GradientTape() as tape:
            # Label guessing
            x_u = tf.reshape(x_u, [s_u[0] * s_u[1], s_u[2], s_u[3], 3])
            pred_u_flat = self.backbone(x_u, training=True)
            pred_u = tf.reshape(pred_u_flat, [s_u[0], self.K, self.num_classes])

            label_u = tf.reduce_mean(pred_u, axis=1)
            label_u = sharpen(label_u, self.T)
            label_u = tf.reshape(tf.tile(tf.expand_dims(label_u, axis=1), [1, self.K, 1]),
                                 [s_u[0] * self.K, self.num_classes])
            label_u = tf.stop_gradient(label_u)

            # Creation of W for mixup
            W = tf.concat([x, x_u], axis=0)
            y = tf.one_hot(y, depth=self.num_classes)
            label_W = tf.concat([y, label_u], axis=0)

            permutation = tf.random.shuffle(tf.range(tf.shape(W)[0]))
            W = tf.gather(W, permutation, axis=0)
            label_W = tf.gather(label_W, permutation, axis=0)

            # Mixup operation
            x_m_sup, y_m_sup = mixup(x, W[:s_u[0]], y, label_W[:s_u[0]], self.lambda_dist)
            x_m_unsup, y_m_unsup = mixup(x_u, W[s_u[0]:], label_u, label_W[s_u[0]:], self.lambda_dist)

            # Calcul of the predictions
            pred_sup = self.backbone(x_m_sup, training=True)
            pred_unsup = self.backbone(x_m_unsup, training=True)

            # Calcul of loss
            loss_sup = self.sup_loss(y_m_sup, pred_sup)
            loss_unsup = tf.reduce_mean(tf.reduce_sum(tf.square(y_m_unsup - pred_unsup), axis=-1))

            final_loss = loss_sup + self.lambda_u * loss_unsup

        # calculate gradients
        train_vars = self.trainable_variables
        gradients = tape.gradient(final_loss, train_vars)

        # update networks
        self.optimizer.apply_gradients(zip(gradients, train_vars))

        # update metrics
        self.unsup_loss_tracker.update_state(loss_unsup)
        self.sup_loss_tracker.update_state(loss_sup)
        self.unsup_accuracy_tracker.update_state(y_m_unsup, pred_unsup)
        self.sup_accuracy_tracker.update_state(y_m_sup, pred_sup)

        return {m.name: m.result() for m in self.metrics}
