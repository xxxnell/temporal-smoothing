import os
import re
import tensorflow as tf
import tensorflow_datasets as tfds

import ops.depth_estimation.imageops as imageops


def dataset(img_size=None, crop_size=None, flip=False):
    dataset_train = tfds.load('nyu_depth_v2', split='train')
    dataset_train = dataset_train.map(lambda dat: (dat['image'], dat['depth']))
    dataset_train = dataset_train.map(lambda xs, ys: preprocessing(xs, ys, img_size))
    dataset_train = dataset_train.map(lambda xs, ys: augment(xs, ys, crop_size, flip=flip))
    dataset_val = tfds.load('nyu_depth_v2', split='validation')
    dataset_val = dataset_val.map(lambda dat: (dat['image'], dat['depth']))
    dataset_val = dataset_val.map(lambda xs, ys: preprocessing(xs, ys, img_size))

    return dataset_train, dataset_val, dataset_val


def dataset_seq(seq_root, img_size, offset, i=None):
    dataset_seq = dataset_raw(seq_root, img_size, i)

    wsize = offset[0] + offset[1] + 1
    dataset_seq = dataset_seq.window(wsize, shift=1).flat_map(
        lambda xs, ys: tf.data.Dataset.zip((xs.batch(wsize), ys.batch(wsize))))
    dataset_seq = dataset_seq.filter(lambda xs, ys: len(ys) >= wsize)
    dataset_seq = dataset_seq.map(lambda xs, ys: (xs, ys[offset[0]]))
    return dataset_seq


def dataset_raw(seq_root, img_size, i=None):
    xs_paths, ys_paths = nyud_subpaths(seq_root, i)

    xs = [imageops.read_color_image(xs_path) for xs_path in xs_paths]
    dataset_xs = tf.data.Dataset.from_tensor_slices(xs)
    ys = [imageops.read_depth_image(ys_path) for ys_path in ys_paths]
    dataset_ys = tf.data.Dataset.from_tensor_slices(ys)

    dataset = tf.data.Dataset.zip((dataset_xs, dataset_ys))
    dataset = dataset.map(lambda x, y: preprocessing(x, y, img_size))
    return dataset


def preprocessing(x, y, img_size):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.float32)
    y = tf.expand_dims(y, axis=-1)

    NEAREST_NEIGHBOR = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    x = tf.image.resize(x, img_size, method=NEAREST_NEIGHBOR)
    y = tf.image.resize(y, img_size, method=NEAREST_NEIGHBOR)

    return x, y


def augment(xs, ys, crop_size=None, flip=False):
    if crop_size is not None:
        crop_size = list(crop_size)
        xs, ys = concat_op(lambda concat: tf.image.random_crop(concat, crop_size + [4]), xs, ys)
    if flip:
        xs, ys = concat_op(lambda concat: tf.image.random_flip_left_right(concat), xs, ys)
    return xs, ys


def concat_op(op, xs, ys):
    concat = tf.concat([xs, ys], axis=-1)
    concat = op(concat)
    xs, ys = tf.split(concat, [xs.shape[-1], -1], axis=-1)

    return xs, ys


def nyud_subpaths(seq_root, i=None):
    xs_paths, ys_paths = [], []
    groups = os.listdir(seq_root)
    groups = [group for group in groups if valid_nyud_group(group)]
    groups = groups if i is None else groups[i:i+1]
    for group in groups:
        filenames = sorted(os.listdir('%s/%s' % (seq_root, group)))
        for filename in filenames:
            if find_nyud_rgb(filename) is not None:
                xs_paths.append('%s/%s/%s' % (seq_root, group, filename))
            if find_nyud_depth(filename) is not None:
                ys_paths.append('%s/%s/%s' % (seq_root, group, filename))
    return xs_paths, ys_paths


def valid_nyud_group(group):
    regex = r'[a-z_]+_[0-9]+[a-z]*'
    pattern = re.compile(regex)
    return pattern.match(group)


def find_nyud_rgb(filename):
    regex = r'(._)*r-([0-9]+.[0-9]+-[0-9]+).ppm'
    elems = re.compile(regex).findall(filename)
    nyud_id = elems[0][1] if elems and not elems[0][0] else None
    return nyud_id


def find_nyud_depth(filename):
    regex = r'(._)*d-([0-9]+.[0-9]+-[0-9]+).pgm'
    elems = re.compile(regex).findall(filename)
    nyud_id = elems[0][1] if elems and not elems[0][0] else None
    return nyud_id
