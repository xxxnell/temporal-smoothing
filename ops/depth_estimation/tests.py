import time
import math
import numpy as np
import tensorflow as tf


def test(predict, dataset,
         batch_size=3, ys_mask=None, cutoffs=(0.0, 0.9), verbose=True, period=10):
    """
    :return: NLL, RMSE, Abs Rel, Sq Rel, n_tot, n_𝛿1, n_𝛿2, n_𝛿3
    """
    predict_times = []
    metrics_mean = [tf.keras.metrics.Mean() for _ in range(8)]

    dataset = dataset.batch(batch_size).enumerate().prefetch(tf.data.experimental.AUTOTUNE)
    for step, (xs, ys) in dataset:
        batch_time = time.time()
        ys_pred = predict(xs)
        predict_times.append(time.time() - batch_time)

        mask = ys_mask(xs, ys)
        ys = tf.boolean_mask(ys, mask)
        ys_pred = tf.boolean_mask(ys_pred, mask)

        metrics = mde_metrics(ys, ys_pred)
        _ = [mean(metric) for mean, metric in zip(metrics_mean, metrics)]

        metrics_str = [
            "Time: %.3f ± %.3f ms" % (np.mean(predict_times) * 1e3, np.std(predict_times) * 1e3),
            "NLL: %.4f" % metrics_mean[0].result(),
            'RMSE: %.3f' % tf.math.sqrt(metrics_mean[1].result()),
            'Abs Rel: %.3f' % metrics_mean[2].result(),
            'Sq Rel: %.3f' % metrics_mean[3].result(),
            '𝛿1: %.3f' % (metrics_mean[5].result() / metrics_mean[4].result()),
            '𝛿2: %.3f' % (metrics_mean[6].result() / metrics_mean[4].result()),
            '𝛿3: %.3f' % (metrics_mean[7].result() / metrics_mean[4].result()),
        ]
        if verbose and int(step + 1) % period is 0:
            print('%d Steps, %s' % (int(step + 1), ', '.join(metrics_str)))

    print(', '.join(metrics_str))

    n_tot = metrics_mean[4].result()
    return metrics_mean[0].result(), metrics_mean[1].result(), \
           metrics_mean[2].result(), metrics_mean[3].result(), \
           metrics_mean[5].result() / n_tot, metrics_mean[6].result() / n_tot, metrics_mean[7].result() / n_tot


def mde_metrics(ys, ys_pred):
    """
    :return: NLL, MSE, Abs Rel, Sq Rel, n_tot, n_𝛿1, n_𝛿2, n_𝛿3
    """
    ys_pred_mean, ys_pred_var = tf.split(ys_pred, [1, 1], axis=-1)

    # NLL
    nll = 0.5 * tf.math.log(2 * math.pi)
    nll = nll + 0.5 * tf.math.log(ys_pred_var)
    nll = nll + 0.5 * tf.math.square(ys - ys_pred_mean) / ys_pred_var

    # RMSE
    mse = tf.math.square(ys - ys_pred_mean)

    # Abs Rel
    abs_rel = tf.math.abs(ys - ys_pred_mean) / ys

    # Sq Rel
    sq_rel = tf.math.square(ys - ys_pred_mean) / ys

    # Accuracy
    acc = tf.stack([ys / ys_pred_mean, ys_pred_mean / ys], axis=-1)
    acc = tf.math.reduce_max(acc, axis=-1)

    n_tot = tf.math.count_nonzero(acc > -np.inf)
    n_1 = tf.math.count_nonzero(acc < 1.25 ** 1)
    n_2 = tf.math.count_nonzero(acc < 1.25 ** 2)
    n_3 = tf.math.count_nonzero(acc < 1.25 ** 3)

    return nll, mse, abs_rel, sq_rel, n_tot, n_1, n_2, n_3


def test_vanilla(model, dataset, batch_size=3,
                 var=0.5, edge=None, cutoffs=(0.0, 0.9), verbose=True, period=10):
    return test(lambda xs: predict_vanilla(model, xs, var=var), dataset, batch_size=batch_size,
                ys_mask=lambda xs, ys: ys_mask(xs, ys, edge), cutoffs=cutoffs, verbose=verbose, period=period)


def test_sampling(model, n_ff, dataset, batch_size=3,
                  var=0.5, edge=None, cutoffs=(0.0, 0.9), verbose=True, period=10):
    return test(lambda xs: predict_sampling(model, xs, n_ff, var=var), dataset, batch_size=batch_size,
                ys_mask=lambda xs, ys: ys_mask(xs, ys, edge), cutoffs=cutoffs, verbose=verbose, period=period)


def predict_vanilla(model, xs, var=0.5):
    ys_pred = model(xs)
    ys_pred = tf.nn.relu(ys_pred)

    ys_pred_mean = ys_pred
    ys_pred_var = tf.ones(ys_pred.shape[:3] + [1]) * var
    ys_pred = tf.concat([ys_pred_mean, ys_pred_var], axis=-1)
    return ys_pred


def predict_sampling(model, xs, n_ff, var=0.5):
    ys_pred = tf.stack([model(xs) for _ in range(n_ff)], axis=-1)
    ys_pred = tf.nn.relu(ys_pred)

    ys_pred_mean = tf.math.reduce_mean(ys_pred, axis=-1)
    ys_pred_var = tf.math.reduce_variance(ys_pred, axis=-1)
    ys_pred_var = ys_pred_var + tf.ones(ys_pred.shape[:3] + [1]) * var
    ys_pred = tf.concat([ys_pred_mean, ys_pred_var], axis=-1)
    return ys_pred


def ys_mask(xs, ys, edge=None):
    # Void mask
    y_min, y_max = 0.0, tf.reduce_max(ys, axis=[1, 2], keepdims=True)
    mask = tf.math.logical_and(ys > y_min, ys < y_max)
    mask = tf.squeeze(mask, axis=-1)

    # Edge mask
    if edge is not None:
        xs_edge = tf.image.sobel_edges(tf.image.rgb_to_grayscale(xs))
        xs_edge = tf.squeeze(xs_edge, axis=3)
        xs_edge = tf.sqrt(tf.reduce_sum(tf.math.square(xs_edge), axis=-1))
        mask = tf.math.logical_and(mask, xs_edge > edge)

    return mask
