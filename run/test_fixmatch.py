import tensorflow as tf
import argparse
import json
import os

from fixmatch.data.dataset_builder import dataset_builder_test
from fixmatch.data.transform import create_supervised_dataset
from fixmatch.networks.wide_resnet import WideResNet
from fixmatch.models import fixmatch

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
    'scheduler_params': {'learning_rate': 0.03, 'decay_steps': 10000},
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

sup_ds, params_data = dataset_builder_test(params['dataset'], split='test')

sup_ds = create_supervised_dataset(sup_ds, params['batch_size'], training=True)

backbone = WideResNet(num_classes=params['num_classes'], depth=params['backbone_depth'], width=params['backbone_width'])
model = fixmatch.Fixmatch(backbone, params['tau'], params['lambda_u'])

latest = tf.train.latest_checkpoint(os.path.join(params['exp_dir'], "checkpoints"))
print(latest)
model.load_weights(latest)

backbone.compile(loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy']) # , run_eagerly=True)

steps = params_data['num_data_test'] // params['batch_size']

backbone.evaluate(sup_ds, steps=steps)