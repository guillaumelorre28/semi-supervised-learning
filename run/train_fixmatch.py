import tensorflow as tf
import tensorflow_addons as tfa
import argparse
import json
import os

from fixmatch.data.dataset_builder import dataset_builder_semisup, dataset_builder_test
from fixmatch.data.transform import create_unsupervised_dataset, create_supervised_dataset
from fixmatch.networks.wide_resnet import WideResNet
from fixmatch.models import fixmatch
from fixmatch.training.learning_rate_scheduler import learning_rate_scheduler
from fixmatch.training.optimizer_builder import optimizer_builder, optimizer_builder_wd

parser = argparse.ArgumentParser()

parser.add_argument(
    '--params_file', type=str, default='',
    help='The name of the .json parameters file to load.')

FLAGS = parser.parse_args()

params = {
    'dataset': 'cifar10',
    'percentage_sup': 10,
    'mu': 1,
    'backbone_depth': 28,
    'backbone_width': 2,
    'image_size': 32,
    'augmentation_type': 'autoaugment',
    'exp_dir': '/media/guillaume/Data/logs/semi_supervised/fixmatch/test_cifar10',
    'num_epochs': 100,
    'batch_size': 64,
    'scheduler_type': 'CONSTANT',
    'scheduler_params': {'learning_rate': 0.03, 'decay_steps': 100000},
    'optimizer_type': 'SGD',
    'optimizer_params': {'momentum': 0.9},
    'num_classes': 10,
    'tau': 0.8,
    'lambda_u': 1.0
}

if FLAGS.params_file != '':
    with open(FLAGS.params_file) as json_file:
        data_params = json.load(json_file)
        params.update(data_params)

sup_ds, unsup_ds, params_data = dataset_builder_semisup(params['dataset'], params['percentage_sup'])

sup_ds = create_supervised_dataset(sup_ds, params['batch_size'], training=True)

unsup_ds = create_unsupervised_dataset(unsup_ds, params['mu'] * params['batch_size'],
                                       strong_augmentation_type=params['augmentation_type'])

combined_ds = tf.data.Dataset.zip((sup_ds, unsup_ds))

test_ds, params_data_test = dataset_builder_test(params['dataset'], "test")
test_ds = create_supervised_dataset(test_ds, params['batch_size'], training=False)

learning_rate = learning_rate_scheduler(params['scheduler_type'], params['scheduler_params'])
weight_decay = 0.0001 # learning_rate_scheduler(params['scheduler_type'], params['weight_decay_params'])

optimizer = optimizer_builder_wd(params['optimizer_type'], learning_rate, weight_decay, params['optimizer_params'])
# optimizer = optimizer_builder(params['optimizer_type'], learning_rate, params['optimizer_params'])
optimizer_ema = tfa.optimizers.MovingAverage(optimizer, decay=0.999)


backbone = WideResNet(num_classes=params['num_classes'], depth=params['backbone_depth'], width=params['backbone_width'])
model = fixmatch.Fixmatch(backbone, params['tau'], params['lambda_u'])

model.compile(optimizer=optimizer_ema) # , run_eagerly=True)

steps_per_epoch = params_data['num_data_sup'] // params['batch_size']
validation_steps = params_data_test['num_data_test'] // params['batch_size']

tensorboard_cb = tf.keras.callbacks.TensorBoard(params['exp_dir'])
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(params['exp_dir'], 'checkpoints/weights.{epoch:02d}.ckpt'),
    save_weights_only=True
)

model.fit(
    combined_ds,
    epochs=params['num_epochs'],
    initial_epoch=0,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_ds,
    validation_steps=validation_steps,
    callbacks=[tensorboard_cb, checkpoint_cb]
)






