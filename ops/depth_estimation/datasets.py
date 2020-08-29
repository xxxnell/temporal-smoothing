import tensorflow as tf
import tensorflow_datasets as tfds


def dataset(img_size=None, crop_size=None, flip=False):
    dataset_train = tfds.load('nyu_depth_v2', split='train')
    dataset_train = dataset_train.map(lambda dat: preprocessing(dat, img_size))
    dataset_train = dataset_train.map(lambda xs, ys: augment(xs, ys, crop_size, flip=flip))
    dataset_val = tfds.load('nyu_depth_v2', split='validation')
    dataset_val = dataset_val.map(lambda dat: preprocessing(dat, img_size))

    return dataset_train, dataset_val, dataset_val


def preprocessing(dat, img_size):
    xs, ys = dat['image'], dat['depth']
    xs = tf.cast(xs, tf.float32) / 255.0
    ys = tf.cast(ys, tf.float32)
    ys = tf.expand_dims(ys, axis=-1)

    NEAREST_NEIGHBOR = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    xs = tf.image.resize(xs, img_size, method=NEAREST_NEIGHBOR)
    ys = tf.image.resize(ys, img_size, method=NEAREST_NEIGHBOR)

    return xs, ys


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
