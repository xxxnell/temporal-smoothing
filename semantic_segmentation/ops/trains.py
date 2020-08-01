import tensorflow as tf


def train_step(optimizer, model, xs, ys, num_classes, class_weights, temp=1.0):
    with tf.GradientTape() as tape:
        logits = model(xs, training=True)

        mask = ys >= 0
        ys = tf.boolean_mask(ys, mask)
        logits = tf.boolean_mask(logits, mask)

        ys = tf.one_hot(ys, num_classes)
        nll = tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=logits)
        weights = tf.reduce_sum(class_weights * ys, axis=-1)
        loss = nll * weights

        reg = sum(model.losses)
        loss = loss + temp * reg

    gradients = tape.gradient(loss, model.trainable_variables)
    grads_and_vars = zip(gradients, model.trainable_variables)
    if not any([tf.reduce_any(tf.math.is_nan(g)) for g in gradients if g is not None]):
        optimizer.apply_gradients(grads_and_vars)
    else:
        raise ValueError("NaN occurs.")
    return loss, nll


def train_epoch(optimizer, model, dataset, num_classes, class_weights, batch_size=3, buffer_size=4096, temp=1.0):
    loss_metric = tf.keras.metrics.Mean(name='train_epoch_loss')
    nll_metric = tf.keras.metrics.Mean(name='train_epoch_nll')
    
    dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    for xs, ys in dataset:
        loss, nll = train_step(optimizer, model, xs, ys, num_classes, class_weights, temp)
        loss_metric(loss), nll_metric(nll)
    return loss_metric.result(), nll_metric.result()
