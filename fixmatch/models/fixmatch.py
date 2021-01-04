import tensorflow as tf

layers = tf.keras.layers


class Fixmatch(tf.keras.Model):

    def __init__(self, backbone, tau, lambda_u, weight_decay=0.0005):

        super(Fixmatch, self).__init__()

        self.backbone = backbone
        self.tau = tau
        self.lambda_u = lambda_u
        self.weight_decay = weight_decay

        self.unsup_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        self.sup_loss = tf.keras.losses.SparseCategoricalCrossentropy()

        self.unsup_loss_tracker = tf.keras.metrics.Mean(name="unsup_loss")
        self.sup_loss_tracker = tf.keras.metrics.Mean(name="sup_loss")

        self.unsup_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
                name="unsup_accuracy")
        self.sup_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="sup_accuracy")

        self.test_loss_tracker = tf.keras.metrics.Mean(name="test_loss")
        self.test_accuracy_tracker = tf.metrics.SparseCategoricalAccuracy(
            name="test_accuracy")

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
    def metrics_test(self):
        _metrics_test = [
            self.test_loss_tracker,
            self.test_accuracy_tracker
        ]

        return _metrics_test

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

            mask = tf.cast(tf.greater(tf.reduce_max(pred_x_w, axis=-1), self.tau), tf.float32)

            labels = tf.argmax(pred_x_w, axis=-1)
            labels = tf.stop_gradient(labels)

            loss_unsup = self.unsup_loss(labels, pred_x_s, sample_weight=mask) / (tf.reduce_sum(mask) + 1.)

            final_loss = loss_sup + self.lambda_u * loss_unsup

            # if self.weight_decay:
            #     wd_loss = 0
            #     for var in self.trainable_variables:
            #         if 'kernel' in var.name:
            #             wd_loss += self.weight_decay * tf.nn.l2_loss(var)
            #     final_loss += wd_loss

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

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self.backbone(x, training=False)
        # Updates the metrics tracking the loss
        loss_test = self.sup_loss(y, y_pred)
        self.test_loss_tracker.update_state(loss_test)
        # Update the metrics.
        self.test_accuracy_tracker.update_state(y, y_pred)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics_test}

