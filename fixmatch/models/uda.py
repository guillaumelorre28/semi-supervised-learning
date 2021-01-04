import tensorflow as tf
from fixmatch.utils import sharpen_uda

layers = tf.keras.layers


class UDA(tf.keras.Model):

    def __init__(self, backbone, beta, tau, lambda_u):

        super(UDA, self).__init__()

        self.backbone = backbone
        self.beta = beta
        self.tau = tau
        self.lambda_u = lambda_u

        self.unsup_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.sup_loss = tf.keras.losses.SparseCategoricalCrossentropy()

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
        x_w, x_s = inputs_unsup

        with tf.GradientTape() as tape:

            pred_x = self.backbone(x, training=True)
            pred_x_w = self.backbone(x_w, training=True)
            pred_x_s = self.backbone(x_s, training=True)

            loss_sup = self.sup_loss(y, pred_x)

            mask = tf.cast(tf.greater(tf.reduce_max(pred_x_w, axis=-1), self.beta), tf.float32)

            labels = sharpen_uda(pred_x_w, self.tau)
            labels = tf.stop_gradient(labels)

            loss_unsup = self.unsup_loss(labels, pred_x_s, sample_weight=mask)

            final_loss = loss_sup + self.lambda_u * loss_unsup

        # calculate gradients
        train_vars = self.trainable_variables
        gradients = tape.gradient(final_loss, train_vars)

        # update networks
        self.optimizer.apply_gradients(zip(gradients, train_vars))

        # update metrics
        self.unsup_loss_tracker.update_state(loss_unsup)
        self.sup_loss_tracker.update_state(loss_sup)
        self.unsup_accuracy_tracker.update_state(labels, pred_x_s, sample_weight=mask)
        self.sup_accuracy_tracker.update_state(y, pred_x)

        return {m.name: m.result() for m in self.metrics}
