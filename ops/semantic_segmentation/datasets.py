import os
import re
from collections import namedtuple
import tensorflow as tf
import numpy as np
import ops.semantic_segmentation.imageops as imageops

Label = namedtuple('Label', [
    # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class
    'name',

    # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.
    'id',

    # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!
    'trainId',

    # The name of the category that this label belongs to
    'category',

    # The ID of this category. Used to create ground truth images
    # on category level.
    'categoryId',

    # Whether this label distinguishes between single instances or not
    'hasInstances',

    # Whether pixels having this class as ground truth label are ignored
    'ignore',

    # The color of this label
    'color',
])

camvid_labels = [
    # name, id, trainId, category, catId, hasInstances, ignore, color
    Label('Void', None, None, 'Void', 0, False, True, (0, 0, 0)),

    Label('Sky', None, None, 'Sky', 1, False, False, (128, 128, 128)),

    Label('Bridge', None, None, 'Building', 2, False, False, (0, 128, 64)),
    Label('Building', None, None, 'Building', 2, False, False, (128, 0, 0)),
    Label('Wall', None, None, 'Building', 2, False, False, (64, 192, 0)),
    Label('Tunnel', None, None, 'Building', 2, False, False, (64, 0, 64)),
    Label('Archway', None, None, 'Building', 2, False, False, (192, 0, 128)),

    Label('Column_Pole', None, None, 'Pole', 3, False, False, (192, 192, 128)),
    Label('TrafficCone', None, None, 'Pole', 3, False, False, (0, 0, 64)),

    Label('Road', None, None, 'Road', 4, False, False, (128, 64, 128)),
    Label('LaneMkgsDriv', None, None, 'Road', 4, False, False, (128, 0, 192)),
    Label('LaneMkgsNonDriv', None, None, 'Road', 4, False, False, (192, 0, 64)),

    Label('Sidewalk', None, None, 'Pavement', 5, False, False, (0, 0, 192)),
    Label('ParkingBlock', None, None, 'Pavement', 5, False, False, (64, 192, 128)),
    Label('RoadShoulder', None, None, 'Pavement', 5, False, False, (128, 128, 192)),

    Label('Tree', None, None, 'Tree', 6, False, False, (128, 128, 0)),
    Label('VegetationMisc', None, None, 'Tree', 6, False, False, (192, 192, 0)),

    Label('SignSymbol', None, None, 'SignSymbol', 7, False, False, (192, 128, 128)),
    Label('Misc_Text', None, None, 'SignSymbol', 7, False, False, (128, 128, 64)),
    Label('TrafficLight', None, None, 'SignSymbol', 7, False, False, (0, 64, 64)),

    Label('Fence', None, None, 'Fence', 8, False, False, (64, 64, 128)),

    Label('Car', None, None, 'Car', 9, False, False, (64, 0, 128)),
    Label('SUVPickupTruck', None, None, 'Car', 9, False, False, (64, 128, 192)),
    Label('Train', None, None, 'Car', 9, False, False, (192, 64, 128)),
    Label('Truck_Bus', None, None, 'Car', 9, False, False, (192, 128, 192)),
    Label('OtherMoving', None, None, 'Car', 9, False, False, (128, 64, 64)),

    Label('Pedestrian', None, None, 'Pedestrian', 10, False, False, (64, 64, 0)),
    Label('Child', None, None, 'Pedestrian', 10, False, False, (192, 128, 64)),
    Label('CartLuggagePram', None, None, 'Pedestrian', 10, False, False, (64, 0, 192)),
    Label('Animal', None, None, 'Pedestrian', 10, False, False, (64, 128, 64)),

    Label('Bicyclist', None, None, 'Bicyclist', 11, False, False, (0, 128, 192)),
    Label('MotorcycleScooter', None, None, 'Bicyclist', 11, False, False, (192, 0, 192)),
]

cityscape_labels = [
    # name, id, trainId, category, catId, hasInstances, ignore, color
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),

    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),

    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),

    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),

    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),

    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),

    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),

    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]


def dataset(name='camvid', dataset_root=None, img_size=None, crop_size=None, flip=False, cache=True):
    if name in ['camvid', 'CamVid', 'camvid-11']:
        if dataset_root is None:
            dataset_root = 'datasets/camvid'
        if img_size is None:
            img_size = 720 // 2, 960 // 2
        if crop_size is None:
            crop_size = 720 // 2, 960 // 2
        dataset_paths = camvid_paths(dataset_root)
        train_img_q, train_label_q, val_img_q, val_label_q, test_img_q, test_label_q = dataset_paths
    elif name in ['camvid-31']:
        if dataset_root is None:
            dataset_root = 'datasets/camvid'
        if img_size is None:
            img_size = 720 // 2, 960 // 2
        if crop_size is None:
            crop_size = 720 // 2, 960 // 2
        dataset_paths = camvid_paths(dataset_root)
        train_img_q, train_label_q, val_img_q, val_label_q, test_img_q, test_label_q = dataset_paths
    elif name in ['cityscape', 'CityScape']:
        if dataset_root is None:
            dataset_root = 'datasets/cityscape'
        if img_size is None:
            img_size = 1024 // 2, 2048 // 2
        if crop_size is None:
            crop_size = 480, 560
        dataset_paths = cityscape_paths(dataset_root)
        train_img_q, train_label_q, val_img_q, val_label_q, _, _ = dataset_paths
        test_img_q, test_label_q = val_img_q, val_label_q
    else:
        raise ValueError('%s is not allowded.' % name)

    cols = colors(name)
    dataset_train = tf.data.Dataset.zip(
        (images_from_paths(train_img_q, img_size), labels_from_paths(train_label_q, cols, img_size)))
    dataset_train = dataset_train.cache() if cache else dataset_train
    dataset_train = dataset_train.map(lambda x, y: augment(x, y, crop_size, flip))
    dataset_val = tf.data.Dataset.zip(
        (images_from_paths(val_img_q, img_size), labels_from_paths(val_label_q, cols, img_size)))
    dataset_val = dataset_val.cache() if cache else dataset_val
    dataset_test = tf.data.Dataset.zip(
        (images_from_paths(test_img_q, img_size), labels_from_paths(test_label_q, cols, img_size)))

    return dataset_train, dataset_val, dataset_test


def dataset_seq(name='camvid', dataset_root=None, seq_root=None, img_size=None, offset=(25, 0)):
    if name in ['camvid', 'CamVid', 'camvid-11']:
        if dataset_root is None:
            dataset_root = 'datasets/camvid'
        if seq_root is None:
            seq_root = 'F:/research/dataset/camvid/seq'
        if img_size is None:
            img_size = 720 // 2, 960 // 2
        img_path, label_path = camvid_seq_paths(dataset_root, seq_root)
    elif name in ['camvid-31']:
        if dataset_root is None:
            dataset_root = 'datasets/camvid'
        if seq_root is None:
            seq_root = 'F:/research/dataset/camvid/seq'
        if img_size is None:
            img_size = 720 // 2, 960 // 2
        img_path, label_path = camvid_seq_paths(dataset_root, seq_root)
    elif name in ['cityscape', 'CityScape']:
        if dataset_root is None:
            dataset_root = 'datasets/cityscape'
        if seq_root is None:
            seq_root = 'F:/research/dataset/cityscape'
        if img_size is None:
            img_size = 1024 // 2, 2048 // 2
        _, _, img_path, label_path, _, _ = cityscape_seq_paths(dataset_root, seq_root)
    else:
        raise ValueError('%s is not allowded.' % name)

    cols = colors(name)
    dataset_img = tf.data.Dataset.from_tensor_slices(img_path)
    dataset_label = tf.data.Dataset.from_tensor_slices([str(path) for path in label_path])
    dataset = tf.data.Dataset.zip((dataset_img, dataset_label))
    wsize = offset[0] + offset[1] + 1
    dataset = dataset.window(wsize, shift=1).flat_map(
        lambda xs, ys: tf.data.Dataset.zip((xs.batch(wsize), ys.batch(wsize))))
    dataset = dataset.filter(lambda imgs, labels: len(labels) >= wsize)
    dataset = dataset.filter(lambda imgs, labels: tf.math.not_equal(labels[offset[0]], str(None)))
    dataset = dataset.flat_map(lambda imgs, labels: read_seq(imgs, labels, img_size, cols, offset))

    return dataset


def dataset_imgseq(name='camvid', dataset_root=None, seq_root=None, img_size=None, skip=0):
    if name in ['camvid', 'CamVid', 'camvid-11']:
        if dataset_root is None:
            dataset_root = 'datasets/camvid'
        if seq_root is None:
            seq_root = 'F:/research/dataset/camvid/seq'
        if img_size is None:
            img_size = 720 // 2, 960 // 2
        img_path, label_path = camvid_seq_paths(dataset_root, seq_root)
    elif name in ['camvid-31']:
        if dataset_root is None:
            dataset_root = 'datasets/camvid'
        if seq_root is None:
            seq_root = 'F:/research/dataset/camvid/seq'
        if img_size is None:
            img_size = 720 // 2, 960 // 2
        img_path, label_path = camvid_seq_paths(dataset_root, seq_root)
    elif name in ['cityscape', 'CityScape']:
        if dataset_root is None:
            dataset_root = 'datasets/cityscape'
        if seq_root is None:
            seq_root = 'F:/research/dataset/cityscape'
        if img_size is None:
            img_size = 1024 // 2, 2048 // 2
        _, _, img_path, label_path, _, _ = cityscape_seq_paths(dataset_root, seq_root)
    else:
        raise ValueError('%s is not allowded.' % name)

    dataset = tf.data.Dataset.from_tensor_slices(img_path)
    dataset = dataset.skip(skip)
    dataset = dataset.flat_map(lambda img_path: images_from_paths([img_path], img_size))

    return dataset


def read_seq(img_paths, label_paths, img_size, colors, offset):
    wsize = offset[0] + offset[1] + 1
    images = images_from_paths(img_paths, img_size)
    labels = labels_from_paths(label_paths[offset[0]:], colors, img_size)
    dataset = tf.data.Dataset.zip((images.batch(wsize), labels))
    return dataset


def images_from_paths(img_paths, img_size=None):
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    dataset = dataset.map(lambda path: read_image(path, img_size))
    return dataset


def read_image(image_path, img_size=None):
    image = imageops.read(image_path)
    if img_size is not None:
        NEAREST_NEIGHBOR = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        image = tf.image.resize(image, img_size, method=NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32) / 255.0

    return image


def labels_from_paths(label_paths, colors, img_size=None):
    dataset = tf.data.Dataset.from_tensor_slices(label_paths)
    dataset = dataset.filter(lambda path: tf.math.not_equal(path, str(None)))
    dataset = dataset.map(lambda path: read_label(path, colors, img_size))
    return dataset


def read_label(label_path, colors, img_size=None):
    label = imageops.read(label_path)
    if img_size is not None:
        NEAREST_NEIGHBOR = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        label = tf.image.resize(label, img_size, method=NEAREST_NEIGHBOR)
    label = imageops.from_color(label, colors)

    return label


def augment(images, labels, crop_size=None, flip=False):
    if crop_size is not None:
        crop_size = list(crop_size)
        images, labels = concat_op(lambda concat: tf.image.random_crop(concat, crop_size + [4]), images, labels)
    if flip:
        images, labels = concat_op(lambda concat: tf.image.random_flip_left_right(concat), images, labels)
    return images, labels


def concat_op(op, images, labels):
    labels = tf.cast(tf.expand_dims(labels, axis=-1), tf.float32)
    concat = tf.concat([images, labels], axis=-1)
    concat = op(concat)
    images, labels = tf.split(concat, [images.shape[-1], -1], axis=-1)

    labels = tf.cast(labels, tf.int32)
    labels = tf.squeeze(labels, axis=-1)
    return images, labels


def colors(name='camvid'):
    if name in ['camvid', 'CamVid', 'camvid-11']:
        label_infos = camvid_labels
        use_category = True
    elif name in ['camvid-31']:
        label_infos = camvid_labels
        use_category = False
    elif name in ['cityscape', 'CityScape']:
        label_infos = cityscape_labels
        use_category = False
    else:
        raise ValueError('%s is not allowded.' % name)

    if use_category:
        label_infos = [label for label in label_infos if label.categoryId is not None and not label.ignore]
        index_map = {cat_id: i for i, cat_id in enumerate(set([label.categoryId for label in label_infos]))}
        color_dict = {label.color: index_map[label.categoryId]
                      for label in label_infos
                      if label.categoryId is not None and not label.ignore}
    else:
        label_infos = [label for label in label_infos if not label.ignore]
        color_dict = {label.color: i
                      for i, label in enumerate(label_infos)}

    return color_dict


def camvid_paths(dataset_root):
    """
    File paths of training set and test set of Camvid.

    :param dataset_root: Absolute path of datasets
    :return: train_img_q, train_label_q, val_img_q, val_label_q, test_img_q, test_label_q
    """
    datasets = [], [], [], [], [], []
    types = "train", "train_labels", "val", "val_labels", "test_avp", "test_avp_labels"
    for dataset, typ in zip(datasets, types):
        for filename in sorted(os.listdir("%s/%s" % (dataset_root, typ))):
            dataset.append("%s/%s/%s" % (dataset_root, typ, filename))

    return datasets


def camvid_seq_paths(dataset_root, dataset_seq_root):
    """
    File paths of sequence on CamVid dataset.

    :param dataset_root:
    :param dataset_seq_root:
    :return: path of sequence, "test" label path if exists else none
    """
    test_superpath = "%s/%s" % (dataset_root, "test_labels")

    label_dict = {}
    for filename in sorted(os.listdir(test_superpath)):
        seq, frame, _ = camvid_structure(filename)
        if valid(filename):
            label_dict[(seq, frame)] = "%s/%s" % (test_superpath, filename)

    img_paths, label_paths = [], []
    for filename in sorted(os.listdir("%s" % dataset_seq_root)):
        seq, frame, _ = camvid_structure(filename)
        if valid(filename):
            img_paths.append("%s/%s" % (dataset_seq_root, filename))
            label_paths.append(label_dict.get((seq, frame)))

    return img_paths, label_paths


def camvid_structure(filename):
    """
    :param filename:
    :return: seq id, frame number, type
    """
    regex = r"([0-9|a-z|A-Z]+)_([0-9|a-z|A-Z]+)(_L)?.png"
    elems = re.compile(regex).findall(filename)[0]
    return elems


def valid(filename):
    return filename[0] != "."


def cityscape_paths(dataset_root):
    """
    File paths of training set and test set of Camvid.

    :param dataset_root: Absolute path of datasets
    :return: train_img_q, train_label_q, val_img_q, val_label_q, test_img_q, test_label_q
    """
    train_img_superpath = "%s/leftImg8bit_trainvaltest/leftImg8bit/train" % dataset_root
    train_label_superpath = "%s/gtFine_trainvaltest/gtFine/train" % dataset_root
    val_img_superpath = "%s/leftImg8bit_trainvaltest/leftImg8bit/val" % dataset_root
    val_label_superpath = "%s/gtFine_trainvaltest/gtFine/val" % dataset_root
    test_img_superpath = "%s/leftImg8bit_trainvaltest/leftImg8bit/test" % dataset_root
    test_label_superpath = "%s/gtFine_trainvaltest/gtFine/test" % dataset_root

    img_paths, label_paths = [], []
    for superpath in (train_img_superpath, val_img_superpath, test_img_superpath):
        img_paths.append(cityscape_subpaths(superpath))
    for superpath in (train_label_superpath, val_label_superpath, test_label_superpath):
        label_subpaths = [path for path in cityscape_subpaths(superpath) if cityscape_structure(path)[4] == "color.png"]
        label_paths.append(label_subpaths)

    return img_paths[0], label_paths[0], img_paths[1], label_paths[1], img_paths[2], label_paths[2]


def cityscape_structure(filename):
    """
    Parse the structure of Cityscape file names.

    :return: city, seq:0>6, frame:0>6, type, ext
    """
    regex = r"([a-zA-Z]+)_(\d+)_(\d+)_([a-zA-Z0-9]+)_*([a-zA-Z]*.[a-zA-Z]+)"
    elems = re.compile(regex).findall(filename)[0]
    return elems


def cityscape_seq_paths(dataset_root, dataset_seq_root):
    """
    File paths of sequence.

    :param dataset_root:
    :param dataset_seq_root:
    :return:
    """
    train_superpath = "%s/leftImg8bit_sequence_trainvaltest/leftImg8bit_sequence/train" % dataset_seq_root
    val_superpath = "%s/leftImg8bit_sequence_trainvaltest/leftImg8bit_sequence/val" % dataset_seq_root
    test_superpath = "%s/leftImg8bit_sequence_trainvaltest/leftImg8bit_sequence/test" % dataset_seq_root
    superpaths = train_superpath, val_superpath, test_superpath

    _, train_label_paths, _, val_label_paths, _, test_label_paths = cityscape_paths(dataset_root)
    label_paths = train_label_paths, val_label_paths, test_label_paths
    label_dicts = {}, {}, {}
    for paths, label_dict in zip(label_paths, label_dicts):
        for path in paths:
            city, seq, frame, _, _ = cityscape_structure(path)
            label_dict[(city, seq, frame)] = path

    img_seq_paths, label_seq_paths = [], []
    for superpath, label_dict in zip(superpaths, label_dicts):
        seq_paths = cityscape_subpaths(superpath)
        img_seq_paths.append(seq_paths)
        label_seq_path = []
        for paths in seq_paths:
            city, seq, frame, _, _ = cityscape_structure(paths)
            label_seq_path.append(label_dict.get((city, seq, frame)))
        label_seq_paths.append(label_seq_path)

    return img_seq_paths[0], label_seq_paths[0], img_seq_paths[1], label_seq_paths[1], img_seq_paths[2], label_seq_paths[2]


def cityscape_subpaths(superpath):
    accs = []
    for group in os.listdir(superpath):
        if valid(group):
            for filename in sorted(os.listdir("%s/%s" % (superpath, group))):
                if valid(filename):
                    accs.append("%s/%s/%s" % (superpath, group, filename))
    return accs


def median_freq_weights(dataset, num_classes, batch_size=3):
    count = tf.zeros([num_classes], dtype=tf.int32)
    for xs, ys in dataset.batch(batch_size):
        ys = tf.reshape(ys, [-1])
        count = count + tf.math.bincount(ys, minlength=num_classes)
    count = tf.cast(count, tf.float32)
    freq = tf.math.divide_no_nan(count, tf.reduce_sum(count))
    weights = tf.math.divide_no_nan(np.median(freq), freq)

    return weights


def memorized_median_freq_weights(name='camvid'):
    if name in ['camvid', 'CamVid', 'camvid-11']:
        weights = tf.constant([
            0.30734012, 0.19833793, 4.7175865,  0.16562003, 0.6806351,  0.42397258,
            4.2133756,  3.256359,   1.,         6.7325764,  9.058633,
        ])
    elif name in ['camvid-31']:
        weights = tf.constant([
            0.027896924, 11.773657, 0.019115845, 0.3273610, 0.0000000, 10.607987,
            0.42964065, 128.60786, 0.016009688, 0.24783362, 44.399693, 0.06718957,
            1.2284949, 2.0446982, 0.041175604, 0.58860964, 3.7385745, 0.6753312,
            1.1540297, 0.29557616, 0.13028075, 0.7452813, 0.000000, 1.,
            1.0005174, 0.66517824, 15.874812, 16.524097, 105.224625, 0.82981503,
            90.09858,
        ])
    elif name in ['cityscape', 'CityScape']:
        weights = tf.constant([
            0.02378161, 0.14417477, 0.03839535, 1.338447, 1.,
            0.70898040, 4.21602300, 1.5896031, 0.05500022, 0.7577104,
            0.21756290, 0.71932510, 6.4886045, 0.12540095, 3.2785587,
            3.72766760, 3.76464400, 8.8897560, 2.1188986
        ])
    else:
        raise ValueError('%s is not allowded.' % name)
    return weights
