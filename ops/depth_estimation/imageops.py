import numpy as np
import tensorflow as tf
from PIL import Image


def read_depth_image(path):
    """Loads an unprocessed depth map extracted from the raw dataset."""
    with open(path, 'rb') as f:
        depth = Image.fromarray(read_pgm(f), mode='I')
        depth = tf.keras.preprocessing.image.img_to_array(depth)
        depth = tf.squeeze(depth, axis=-1)

        # Rel depth -> Abs depth
        depth_params = 351.3, 1092.5
        depth = depth_params[0] / (depth_params[1] - depth)
        depth = tf.nn.relu(depth)

        return depth


def read_color_image(path):
    """Loads an unprocessed color image extracted from the raw dataset."""
    with open(path, 'rb') as f:
        img = Image.fromarray(read_ppm(f), mode='RGB')
        img = tf.keras.preprocessing.image.img_to_array(img, dtype=int)
        img = tf.convert_to_tensor(img)
        return img


def read_pgm(pgm_file):
    """Reads a PGM file from a buffer.
    Returns a numpy array of the appropiate size and dtype.
    """

    # First line contains some image meta-info
    p5, width, height, depth = pgm_file.readline().split()

    # Ensure we're actually reading a P5 file
    assert p5 == b'P5'
    assert depth == b'65535', "Only 16-bit PGM files are supported"

    width, height = int(width), int(height)

    data = np.fromfile(pgm_file, dtype='<u2', count=width * height)

    return data.reshape(height, width).astype(np.uint32)


def read_ppm(ppm_file):
    """Reads a PPM file from a buffer.
    Returns a numpy array of the appropiate size and dtype.
    """

    p6, width, height, depth = ppm_file.readline().split()

    assert p6 == b'P6'
    assert depth == b'255', "Only 8-bit PPM files are supported"

    width, height = int(width), int(height)

    data = np.fromfile(ppm_file, dtype=np.uint8, count=width * height * 3)

    return data.reshape(height, width, 3)

