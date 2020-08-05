import csv
import os
import numpy as np
import tensorflow as tf


def dataset(name, dataset_root, cache=True):
    if name in ['occupancy']:
        mean = [2.06194954e+01, 2.57320976e+01, 1.19519295e+02, 6.06546814e+02, 3.86253325e-03]
        std = [1.0168638e+00, 5.5308995e+00, 1.9474380e+02, 3.1430164e+02, 8.5227750e-04]
        train_paths = ['%s/%s' % (dataset_root, file_name) for file_name in ['datatraining.txt']]
        test_paths = ['%s/%s' % (dataset_root, file_name) for file_name in ['datatest.txt', 'datatest2.txt']]
        dataset_train = dataset_from_paths(occupancy_dataset_from_path, train_paths)
        dataset_test = dataset_from_paths(occupancy_dataset_from_path, test_paths)
    elif name in ['emg']:
        mean = np.array([
            -8.0453456e-06, -9.4565294e-06, -9.7021521e-06, -9.5586129e-06,
            -1.6126460e-05, -1.0878757e-05, -9.2788468e-06, -9.4272409e-06
        ])
        std = np.array([
            0.00017179, 0.00012795, 0.00013959, 0.00022451,
            0.00025246, 0.00018531, 0.00015194, 0.00016413
        ])
        train_paths = ['%s/%02d' % (dataset_root, dataset_no) for dataset_no in range(1, 33)]
        test_paths = ['%s/%02d' % (dataset_root, dataset_no) for dataset_no in range(33, 37)]
        dataset_train = dataset_from_paths(emg_dataset_from_path, train_paths)
        dataset_test = dataset_from_paths(emg_dataset_from_path, test_paths)
    elif name in ['localization']:
        mean = np.array([1.483336, 2.801267, 1.6969863, 0.4177647])
        std = np.array([1.1384217, 0.9257093, 0.4791263, 0.37928084])
        file_paths = ['%s/%s' % (dataset_root, 'ConfLongDemo_JSI.txt')]
        dataset_train = dataset_from_paths(localization_dataset_from_path, file_paths)
        dataset_train, dataset_test = dataset_train.take(int(164859 * 0.9)), dataset_train.skip(int(164859 * 0.9))
    else:
        raise ValueError

    dataset_train = dataset_train.map(lambda xs, ys: ((xs - mean) / std, ys))
    dataset_test = dataset_test.map(lambda xs, ys: ((xs - mean) / std, ys))

    if cache:
        dataset_train = dataset_train.cache()
        dataset_test = dataset_test.cache()

    return dataset_train, dataset_test


def dataset_from_paths(dataset_from_path, paths):
    datasets = [dataset_from_path(path) for path in paths]
    datasets = tf.data.Dataset.from_tensor_slices(datasets)
    datasets = datasets.flat_map(lambda dataset: dataset)
    return datasets


def occupancy_dataset_from_path(path):
    xs, ys = [], []
    with open(path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            xs.append(tf.constant([float(v) for v in row[2: 7]]))
            ys.append(tf.constant(int(row[7])))
    xs = tf.data.Dataset.from_tensor_slices(xs)
    ys = tf.data.Dataset.from_tensor_slices(ys)
    dataset = tf.data.Dataset.zip((xs, ys))
    return dataset


def emg_dataset_from_path(path):
    xs, ys = [], []
    for file_name in os.listdir(path):
        with open('%s/%s' % (path, file_name), 'r') as csvfile:
            file_reader = csv.reader(csvfile, delimiter='\t')
            header = next(file_reader)
            for row in file_reader:
                if len(row) == 10:
                    xs.append(tf.constant([float(v) for v in row[1:9]]))
                    ys.append(tf.constant(int(row[9])))
    xs = tf.data.Dataset.from_tensor_slices(xs)
    ys = tf.data.Dataset.from_tensor_slices(ys)
    dataset = tf.data.Dataset.zip((xs, ys))
    return dataset


def localization_dataset_from_path(path):
    TAG = ["010-000-024-033", "010-000-030-096", "020-000-033-111", "020-000-032-221"]
    ACTIVITY = [
        "walking","falling","lying down","lying","sitting down",
        "sitting","standing up from lying","on all fours","sitting on the ground","standing up from sitting",
        "standing up from sitting on the ground"
    ]
    xs, ys = [], []
    with open(path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            xs.append(tf.constant([TAG.index(row[1])] + [float(v) for v in row[4:7]]))
            ys.append(tf.constant(ACTIVITY.index(row[7])))
    xs = tf.data.Dataset.from_tensor_slices(xs)
    ys = tf.data.Dataset.from_tensor_slices(ys)
    dataset = tf.data.Dataset.zip((xs, ys))
    return dataset


def median_freq_weights(dataset, num_classes, batch_size=3):
    count = tf.zeros([num_classes], dtype=tf.int32)
    for xs, ys in dataset.batch(batch_size):
        ys = tf.reshape(ys, [-1])
        count = count + tf.math.bincount(ys, minlength=num_classes)
    count = tf.cast(count, tf.float32)
    freq = tf.math.divide_no_nan(count, tf.reduce_sum(count))
    weights = tf.math.divide_no_nan(np.median(freq), freq)

    return weights


def sliding_window(dataset, offset):
    wsize = offset[0] + offset[1] + 1
    dataset = dataset.window(wsize, shift=1).flat_map(
        lambda xs, ys: tf.data.Dataset.zip((xs.batch(wsize), ys.batch(wsize))))
    dataset = dataset.filter(lambda xs, ys: len(ys) >= wsize)
    dataset = dataset.map(lambda xs, ys: (xs, ys[offset[0]]))
    return dataset
