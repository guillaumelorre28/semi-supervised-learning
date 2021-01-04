import tensorflow as tf
import tensorflow_addons as tfa
from official.vision.image_classification import augment


def random_translation(image, max_translate=0.125):

    h = tf.cast(tf.shape(image[0]), tf.float32)
    w = tf.cast(tf.shape(image[1]), tf.float32)

    dx = tf.cast(tf.random.uniform([], maxval=h*max_translate), tf.int32)
    dy = tf.cast(tf.random.uniform([], maxval=w*max_translate), tf.int32)

    return tfa.image.translate(image, [dx, dy])


def create_supervised_dataset(
    dataset: tf.data.Dataset,
    batch_size: int = 256,
    w: int = 4,
    training:bool = True)->tf.data.Dataset:

    def _preprocess_train(image):
        image_size = tf.shape(image)
        image = tf.cast(image, tf.float32) / 255.
        image = tf.image.random_flip_left_right(image)
        image = tf.pad(image, [[w] * 2, [w] * 2, [0] * 2], mode='REFLECT')
        image = tf.image.random_crop(image, image_size)
        # image = random_translation(image, max_translate)

        return image

    def _preprocess_test(image):
        image = tf.cast(image, tf.float32) / 255.

        return image

    if training:
        dataset = dataset.shuffle(1024).repeat()
        preprocess_fn = _preprocess_train
    else:
        preprocess_fn = _preprocess_test

    dataset = dataset.map(lambda ex: (preprocess_fn(ex['image']), ex['label']),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def create_unsupervised_dataset(
    dataset: tf.data.Dataset,
    batch_size: int = 256,
    w: int = 4,
    strong_augmentation_type: str = "randaugment")->tf.data.Dataset:

    if strong_augmentation_type == "autoaugment":
        augmenter = augment.AutoAugment()
    if strong_augmentation_type == "randaugment":
        augmenter = augment.RandAugment()

    def _weak_augmentation(image):
        image_size = tf.shape(image)
        image = tf.cast(image, tf.float32) / 255.
        image = tf.image.random_flip_left_right(image)
        image = tf.pad(image, [[w] * 2, [w] * 2, [0] * 2], mode='REFLECT')
        image = tf.image.random_crop(image, image_size)
        # image = random_translation(image, max_translate)

        return image

    def _strong_augmentation(image):
        image = tf.image.random_flip_left_right(image)
        image = augmenter.distort(image)
        image = tf.cast(image, tf.float32) / 255.

        return image

    dataset = dataset.shuffle(1024).repeat()

    dataset = dataset.map(lambda ex: (_weak_augmentation(ex['image']), _strong_augmentation(ex['image'])),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


