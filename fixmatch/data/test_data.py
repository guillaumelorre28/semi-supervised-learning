import tensorflow as tf
from fixmatch.data.dataset_builder import dataset_builder_test
from fixmatch.data.transform import create_supervised_dataset
import matplotlib.pyplot as plt

params = {
    'dataset': 'cifar10',
    'batch_size': 5
}

sup_ds, params_data = dataset_builder_test(params['dataset'], split="train")
test_ds, _ = dataset_builder_test(params['dataset'], split="test")

sup_ds = create_supervised_dataset(sup_ds, params['batch_size'], training=True)
test_ds = create_supervised_dataset(test_ds, params['batch_size'], training=False)

for ex in sup_ds.take(5):

    image = ex[0].numpy()
    plt.imshow(image[0])
    plt.show()

print("Test")

for ex in test_ds.take(5):
    image = ex[0].numpy()
    plt.imshow(image[0])
    plt.show()
