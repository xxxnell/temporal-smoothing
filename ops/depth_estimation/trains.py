import numpy as np
import tensorflow as tf
import ops.depth_estimation.tests as tests


def train_step(optimizer, model, xs, ys, temp=1.0):
    with tf.GradientTape() as tape:
        ys_pred = model(xs, training=True)
        ys_pred = tf.nn.relu(ys_pred)

        mask = tests.ys_mask(xs, ys)
        ys = tf.boolean_mask(ys, mask)
        ys_pred = tf.boolean_mask(ys_pred, mask)

        mse = tf.math.square(ys_pred - ys)
        loss = mse

        reg = sum(model.losses)
        loss = loss + temp * reg

    gradients = tape.gradient(loss, model.trainable_variables)
    grads_and_vars = zip(gradients, model.trainable_variables)
    if not any([tf.reduce_any(tf.math.is_nan(g)) for g in gradients if g is not None]):
        optimizer.apply_gradients(grads_and_vars)
    else:
        raise ValueError("NaN occurs.")
    return loss, mse


def train_epoch(optimizer, model, dataset, batch_size=3, buffer_size=4096, temp=1.0):
    loss_metric = tf.keras.metrics.Mean(name='train_epoch_loss')
    mse_metric = tf.keras.metrics.Mean(name='train_epoch_mse')

    dataset = dataset.shuffle(buffer_size).batch(batch_size).enumerate().prefetch(tf.data.experimental.AUTOTUNE)
    for step, (xs, ys) in dataset:
        loss, mse = train_step(optimizer, model, xs, ys, temp)
        loss_metric(loss), mse_metric(mse)

    return loss_metric.result(), mse_metric.result()

