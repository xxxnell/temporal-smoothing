import numpy as np
import tensorflow as tf


def read(path, channels=3):
    return tf.image.decode_png(tf.io.read_file(path), channels=channels)


def crop_random(images, labels, crop_height, crop_width):
    labels = tf.cast(tf.expand_dims(labels, axis=-1), tf.float32)
    concat = tf.concat([images, labels], axis=-1)
    img_channel_no, channel_no = images.shape[-1], concat.shape[-1]
    concat = tf.image.random_crop(concat, [crop_height, crop_width, channel_no])
    images, labels = tf.split(concat, [img_channel_no, -1], axis=-1)
    labels = tf.cast(labels, tf.int32)
    return images, labels


def from_color(label, colors):
    indexed = []
    for color, index in colors.items():
        valid = tf.cast(tf.math.reduce_all(tf.math.equal(label, color), axis=-1), dtype=tf.int32)
        indexed.append(valid * index)
    indexed = tf.reduce_sum(indexed, axis=0)
    return indexed


def to_color(indexed, colors):
    colors = {i: color for color, i in colors.items()}
    label = []
    for index, color in colors.items():
        valid = tf.cast(tf.math.equal(indexed, index), dtype=tf.int32)
        label.append(tf.tensordot(valid, color, axes=0))
    label = tf.reduce_sum(label, axis=0)
    return label


