import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ops.semantic_segmentation.imageops as imageops
from vqbnn.vqbnn import VQBNN
from vqbnn.och import OCH


def test(predict, dataset, num_classes,
         batch_size=3, ys_mask=None, cutoffs=(0.0, 0.9), bins=np.linspace(0.0, 1.0, 11), verbose=True, period=10):
    predict_times = []
    nll_metric = tf.keras.metrics.SparseCategoricalCrossentropy(name='nll')
    cm_shape = [num_classes, num_classes]
    cms = [[np.zeros(cm_shape), np.zeros(cm_shape)] for _ in range(len(cutoffs))]
    ious, accs, uncs, covs, eces = [], [], [], [], []

    cms_bin = [np.zeros(cm_shape) for _ in range(len(bins) - 1)]
    confs_metric_bin = [tf.keras.metrics.Mean() for _ in range(len(bins) - 1)]
    count_bin, accs_bin, confs_bin, metrics_str = [], [], [], []

    dataset = dataset.batch(batch_size).enumerate().prefetch(tf.data.experimental.AUTOTUNE)
    for step, (xs, ys) in dataset:
        batch_time = time.time()
        ys_pred = predict(xs)
        predict_times.append(time.time() - batch_time)

        mask = ys_mask(xs, ys)
        ys = tf.boolean_mask(ys, mask)
        ys_pred = tf.boolean_mask(ys_pred, mask)

        nll_metric(ys, ys_pred)
        for cutoff, cm_group in zip(cutoffs, cms):
            cm_certain = cm(ys, ys_pred, num_classes, filter_min=cutoff)
            cm_uncertain = cm(ys, ys_pred, num_classes, filter_max=cutoff)
            cm_group[0] = cm_group[0] + cm_certain
            cm_group[1] = cm_group[1] + cm_uncertain
        for i, (start, end) in enumerate(zip(bins, bins[1:])):
            cms_bin[i] = cms_bin[i] + cm(ys, ys_pred, num_classes, filter_min=start, filter_max=end)
            confidence = tf.math.reduce_max(ys_pred, axis=-1)
            condition = tf.logical_and(confidence >= start, confidence < end)
            confs_metric_bin[i](tf.boolean_mask(confidence, condition))

        ious = [miou(cm_certain) for cm_certain, cm_uncertain in cms]
        accs = [gacc(cm_certain) for cm_certain, cm_uncertain in cms]
        uncs = [unconfidence(cm_certain, cm_uncertain) for cm_certain, cm_uncertain in cms]
        covs = [coverage(cm_certain, cm_uncertain) for cm_certain, cm_uncertain in cms]
        count_bin = [np.sum(cm_bin) for cm_bin in cms_bin]
        accs_bin = [gacc(cm_bin) for cm_bin in cms_bin]
        confs_bin = [metric.result() for metric in confs_metric_bin]
        eces = ece(count_bin, accs_bin, confs_bin)

        metrics_str = [
            "Time: %.3f ± %.3f ms" % (np.mean(predict_times) * 1e3, np.std(predict_times) * 1e3),
            "NLL: %.4f" % nll_metric.result(),
            "Cutoffs: " + ", ".join(["%.1f %%" % (cutoff * 100) for cutoff in cutoffs]),
            "IoUs: " + ", ".join(["%.3f %%" % (iou * 100) for iou in ious]),
            "Accs: " + ", ".join(["%.3f %%" % (acc * 100) for acc in accs]),
            "Uncs: " + ", ".join(["%.3f %%" % (unc * 100) for unc in uncs]),
            "Covs: " + ", ".join(["%.3f %%" % (cov * 100) for cov in covs]),
            "ECE: " + "%.3f %%" % (eces * 100),
        ]
        if verbose and int(step + 1) % period is 0:
            print('%d Steps, %s' % (int(step + 1), ', '.join(metrics_str)))

    print(", ".join(metrics_str))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    confidence_histogram(axes[0], count_bin)
    reliability_diagram(axes[1], accs_bin)
    fig.tight_layout()
    calibration_image = imageops.plot_to_image(fig)
    if not verbose:
        plt.close(fig)

    return nll_metric.result(), cms, ious, accs, uncs, covs, \
           count_bin, accs_bin, confs_bin, eces, calibration_image


def ys_mask(xs, ys, edge):
    # Void mask
    mask = ys >= 0

    # Edge mask
    if edge is not None:
        xs_edge = tf.image.sobel_edges(tf.image.rgb_to_grayscale(xs))
        xs_edge = tf.squeeze(xs_edge, axis=3)
        xs_edge = tf.sqrt(tf.reduce_sum(tf.math.square(xs_edge), axis=-1))
        mask = tf.math.logical_and(mask, xs_edge > edge)

    return mask


def ys_mask_seq(xs, ys, edge):
    # Void mask
    mask = ys >= 0

    # Edge mask
    if edge is not None:
        xs_edge = tf.image.sobel_edges(tf.image.rgb_to_grayscale(xs[:, -1, :, :, :]))
        xs_edge = tf.squeeze(xs_edge, axis=3)
        xs_edge = tf.sqrt(tf.reduce_sum(tf.math.square(xs_edge), axis=-1))
        mask = tf.math.logical_and(mask, xs_edge > edge)

    return mask


def test_vanilla(model, dataset, num_classes, batch_size=3,
                 edge=None, cutoffs=(0.0, 0.9), verbose=True, period=10):
    return test(lambda xs: predict_vanilla(model, xs), dataset, num_classes, batch_size=batch_size,
                ys_mask=lambda xs, ys: ys_mask(xs, ys, edge), cutoffs=cutoffs, verbose=verbose, period=period)


def test_temp_scaling(model, temp, dataset, num_classes, batch_size=3,
                      edge=None, cutoffs=(0.0, 0.9), verbose=True, period=10):
    return test(lambda xs: predict_temp_scaling(model, xs, temp), dataset, num_classes, batch_size=batch_size,
                ys_mask=lambda xs, ys: ys_mask(xs, ys, edge), cutoffs=cutoffs, verbose=verbose, period=period)


def test_sampling(model, n_ff, dataset, num_classes, batch_size=3,
                  edge=None, cutoffs=(0.0, 0.9), verbose=True, period=10):
    return test(lambda xs: predict_sampling(model, xs, n_ff), dataset, num_classes, batch_size=batch_size,
                ys_mask=lambda xs, ys: ys_mask(xs, ys, edge), cutoffs=cutoffs, verbose=verbose, period=period)


def test_temporal_smoothing(model, l, dataset, num_classes, batch_size=3,
                            edge=None, cutoffs=(0.0, 0.9), verbose=True, period=10):
    return test(lambda xs: predict_temporal_smoothing(model, xs, l), dataset, num_classes, batch_size=batch_size,
                ys_mask=lambda xs, ys: ys_mask_seq(xs, ys, edge), cutoffs=cutoffs, verbose=verbose, period=period)


def predict_vanilla(model, xs):
    ys_pred = tf.nn.softmax(model(xs), axis=-1)
    return ys_pred


def predict_temp_scaling(model, xs, temp=1.0):
    ys_pred = tf.nn.softmax(model(xs) / temp, axis=-1)
    return ys_pred


def predict_sampling(model, xs, n_ff):
    ys_pred = tf.stack([tf.nn.softmax(model(xs), axis=-1) for _ in range(n_ff)])
    ys_pred = tf.reduce_sum(ys_pred, axis=0) / n_ff
    return ys_pred


def predict_temporal_smoothing(model, xs, l):
    n_ff = xs.shape[1]
    weight = (tf.range(n_ff, dtype=tf.float32) - n_ff) * l
    weight = tf.math.exp(weight)
    weight = weight / tf.reduce_sum(weight)

    xs = tf.einsum("ij...->ji...", xs)  # xs = tf.transpose(xs, perm=[1, 0, ...])
    ys_pred = tf.stack([tf.nn.softmax(model(x_batch), axis=-1) for x_batch in xs])
    ys_pred = tf.tensordot(weight, ys_pred, axes=[0, 0])
    return ys_pred


def predict_vq(vqbnn, xs):
    # img_size = xs.shape[2:]
    # x_dims, y_dims = [img_size[0] * img_size[1] * img_size[2]], [img_size[0] * img_size[1] * 1]
    # och_x = OCH(**och_x_params, dims=x_dims, hash_no=1)
    # och_y = OCH(**och_y_params, dims=y_dims, hash_no=1, ann='argmax')
    # vqbnn = VQBNN(lambda x: model(tf.reshape(x, [1] + img_size)), och_x=och_x, och_y=och_y, posterior=None)

    for x in xs[0]:
        vqbnn.update(x)

    ys_pred = tf.reduce_sum([tf.nn.softmax(c, axis=-1) * w for c, w in vqbnn.och_y.cws()], axis=0)
    return ys_pred


def cm(ys, ys_pred, num_classes, filter_min=0.0, filter_max=1.0):
    """
    Confusion matrix.

    :param ys: [batch_size, height, width]
    :param ys_pred: onehot with shape [batch_size, height, width, num_class]
    :param num_classes: int
    :param filter_min:
    :param filter_max:
    :return: cms for certain and uncertain prediction (shape: [batch_size, num_classes, num_classes])
    """
    ys = tf.reshape(tf.cast(ys, tf.int32), [-1])
    result = tf.reshape(tf.argmax(ys_pred, axis=-1, output_type=tf.int32), [-1])
    confidence = tf.reshape(tf.math.reduce_max(ys_pred, axis=-1), [-1])
    condition = tf.logical_and(confidence >= filter_min, confidence < filter_max)

    k = (ys >= 0) & (ys < num_classes) & condition
    cm = tf.math.bincount(num_classes * ys[k] + result[k], minlength=num_classes ** 2)
    cm = tf.reshape(cm, [num_classes, num_classes])

    return cm


def miou(cm):
    """
    Mean IoU
    """
    weights = np.sum(cm, axis=1)
    weights = [1 if weight > 0 else 0 for weight in weights]
    if np.sum(weights) > 0:
        _miou = np.average(ious(cm), weights=weights)
    else:
        _miou = 0.0
    return _miou


def ious(cm):
    """
    Intersection over unit w.r.t. classes.
    """
    num = np.diag(cm)
    den = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=(den != 0))


def gacc(cm):
    """
    Global accuracy p(accurate). For cm_certain, p(accurate|confident).
    """
    num = np.diag(cm).sum()
    den = np.sum(cm)
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=(den != 0))


def accs(cm):
    """
    Accuracies w.r.t. classes.
    """
    accs = []
    for ii in range(np.shape(cm)[0]):
        if float(np.sum(cm, axis=1)[ii]) == 0:
            acc = 0.0
        else:
            acc = np.diag(cm)[ii] / float(np.sum(cm, axis=1)[ii])
        accs.append(acc)
    return accs


def unconfidence(cm_certain, cm_uncertain):
    """
    p(unconfident|inaccurate)
    """
    inaccurate_certain = np.sum(cm_certain) - np.diag(cm_certain).sum()
    inaccurate_uncertain = np.sum(cm_uncertain) - np.diag(cm_uncertain).sum()

    return inaccurate_uncertain / (inaccurate_certain + inaccurate_uncertain)


def coverage(cm_certain, cm_uncertain):
    return np.sum(cm_certain) / (np.sum(cm_certain) + np.sum(cm_uncertain))


def ece(count_bin, accs_bin, confs_bin):
    count_bin = np.array(count_bin)
    accs_bin = np.array(accs_bin)
    confs_bin = np.array(confs_bin)
    freq = np.nan_to_num(count_bin / sum(count_bin))
    ece_result = sum(np.absolute(accs_bin - confs_bin) * freq)
    return ece_result


def confidence_histogram(ax, count_bin):
    color, ALPHA = "tab:green", 0.8
    centers = np.linspace(0.05, 0.95, 10)
    count_bin = np.array(count_bin)
    freq = count_bin / sum(count_bin)

    ax.bar(centers * 100, freq * 100, width=10, color=color, edgecolor="black", alpha=ALPHA)
    ax.set_xlim(0, 100.0)
    ax.set_ylim(0, 100.0)
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Frequency (%)")


def reliability_diagram(ax, accs_bins, colors='tab:red', mode=0):
    ALPHA, GUIDELINE_STYLE = 0.8, (0, (1, 1))
    guides_x, guides_y = np.linspace(0.0, 1.0, 11), np.linspace(0.0, 1.0, 11)
    centers = np.linspace(0.05, 0.95, 10)
    accs_bins = np.array(accs_bins)
    accs_bins = np.expand_dims(accs_bins, axis=0) if len(accs_bins.shape) < 2 else accs_bins
    colors = [colors] if type(colors) is not list else colors
    colors = colors + [None] * (len(accs_bins) - len(colors))

    ax.plot(guides_x * 100, guides_y * 100, linestyle=GUIDELINE_STYLE, color="black")
    for accs_bin, color in zip(accs_bins, colors):
        if mode is 0:
            ax.bar(centers * 100, accs_bin * 100, width=10, color=color, edgecolor="black", alpha=ALPHA)
        elif mode is 1:
            ax.plot(centers * 100, accs_bin * 100, color=color, marker='o', alpha=ALPHA)
        else:
            raise ValueError('Invalid mode %d.' % mode)

    ax.set_xlim(0, 100.0)
    ax.set_ylim(0, 100.0)
    ax.set_xlabel('Confidence (%)')
    ax.set_ylabel('Accuracy (%)')
