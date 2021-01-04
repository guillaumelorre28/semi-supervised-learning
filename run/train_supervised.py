import tensorflow as tf
import tensorflow_addons as tfa
import argparse
import json
import os

from fixmatch.data.dataset_builder import dataset_builder_test
from fixmatch.data.transform import create_supervised_dataset
from fixmatch.networks.wide_resnet import WideResNet
from fixmatch.training.learning_rate_scheduler import learning_rate_scheduler
from fixmatch.training.optimizer_builder import optimizer_builder, optimizer_builder_wd

parser = argparse.ArgumentParser()

parser.add_argument(
    '--params_file', type=str, default='',
    help='The name of the .json parameters file to load.')

FLAGS = parser.parse_args()

params = {
    'dataset': 'cifar10',
    'backbone_depth': 28,
    'backbone_width': 2,
    'image_size': 32,
    'exp_dir': '/media/guillaume/Data/logs/supervised/fixmatch/test_cifar10',
    'num_epochs': 100,
    'batch_size': 64,
    'scheduler_type': 'COSINE_DECAY',
    'scheduler_params': {'initial_learning_rate': 0.03, 'decay_steps': 10000},
    'weight_decay_params': {'initial_learning_rate': 0.0001, 'decay_steps': 10000},
    'optimizer_type': 'SGD',
    'optimizer_params': {'momentum': 0.9},
    'num_classes': 10
}

if FLAGS.params_file != '':
    with open(FLAGS.params_file) as json_file:
        data_params = json.load(json_file)
        params.update(data_params)

sup_ds, params_data = dataset_builder_test(params['dataset'], split="train")
test_ds, _ = dataset_builder_test(params['dataset'], split="test")

sup_ds = create_supervised_dataset(sup_ds, params['batch_size'], training=True)
test_ds = create_supervised_dataset(test_ds, params['batch_size'], training=False)

learning_rate = learning_rate_scheduler(params['scheduler_type'], params['scheduler_params'])
weight_decay = learning_rate_scheduler(params['scheduler_type'], params['weight_decay_params'])

optimizer = optimizer_builder_wd(params['optimizer_type'], learning_rate, weight_decay, params['optimizer_params'])
optimizer_ema = tfa.optimizers.MovingAverage(optimizer, decay=0.999)

backbone = WideResNet(num_classes=params['num_classes'], depth=params['backbone_depth'], width=params['backbone_width'])

backbone.compile(optimizer=optimizer_ema, loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

steps_per_epoch = params_data['num_data_train'] // params['batch_size']
validation_steps = params_data['num_data_test'] // params['batch_size']

tensorboard_cb = tf.keras.callbacks.TensorBoard(params['exp_dir'])
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(params['exp_dir'], 'checkpoints/weights.{epoch:02d}.ckpt'),
    save_weights_only=True
)

backbone.fit(sup_ds,
             epochs=params['num_epochs'],
             steps_per_epoch=steps_per_epoch,
             validation_data=test_ds,
             validation_steps=validation_steps,
             callbacks=[tensorboard_cb, checkpoint_cb]
             )
