import tensorflow as tf
import tensorflow_addons as tfa


def optimizer_builder(optimizer_type, learning_rate, optimizer_params):

    if optimizer_type == 'Adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate, **optimizer_params)

    elif optimizer_type == 'SGD':
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, **optimizer_params)

    elif optimizer_type == 'LAMB':
        return tfa.optimizers.LAMB(learning_rate, **optimizer_params)

    else:
        raise Exception("Invalid optimizer")


def optimizer_builder_wd(optimizer_type, learning_rate, weight_decay, optimizer_params):

    optimizers_dict = {
        'Adam': tf.keras.optimizers.Adam,
        'SGD': tf.keras.optimizers.SGD,
        'LAMB': tfa.optimizers.LAMB
    }

    base_optimizer = optimizers_dict[optimizer_type]

    optimizer = tfa.optimizers.extend_with_decoupled_weight_decay(base_optimizer)(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        **optimizer_params
    )

    return optimizer


