import tensorflow as tf
import tensorflow_datasets as tfds


def dataset_builder_semisup(dataset_name: str, percentage_sup: int=10):

    dataset_sup, info = tfds.load(dataset_name, split=f"train[:{percentage_sup}%]", with_info= True,
                            data_dir="/media/guillaume/Data/data/tensorflow_datasets")

    dataset_unsup, info_unsup = tfds.load(dataset_name, split=f"train[{percentage_sup}%:]", with_info=True,
                            data_dir="/media/guillaume/Data/data/tensorflow_datasets")

    params_data = {
        'num_data_sup': info.splits['train'].num_examples * percentage_sup / 100,
        'num_data_unsup': info.splits['train'].num_examples * (100-percentage_sup) / 100,
    }

    return dataset_sup, dataset_unsup, params_data


def dataset_builder_test(dataset_name: str, split: str):

    dataset_sup, info = tfds.load(dataset_name, split=split, with_info= True,
                            data_dir="/media/guillaume/Data/data/tensorflow_datasets")
    params_data = {
        'num_data_train': info.splits['train'].num_examples,
        'num_data_test': info.splits['test'].num_examples
    }

    return dataset_sup, params_data
