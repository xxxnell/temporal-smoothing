import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model


class SegNet(Model):

    def __init__(self, num_classes, rate=0.0, name="segnet"):
        super(SegNet, self).__init__(name=name)
        self.num_classes = num_classes
        self.rate = rate
        self.strides = [2, 2]

        # Encoding

        self.conv_1_1 = ConvBlock(64)
        self.conv_1_2 = ConvBlock(64)

        self.conv_2_1 = ConvBlock(128)
        self.conv_2_2 = ConvBlock(128)

        self.conv_3_1 = ConvBlock(256)
        self.conv_3_2 = ConvBlock(256)
        self.conv_3_3 = ConvBlock(256)
        self.dropout_3 = layers.Dropout(self.rate)

        self.conv_4_1 = ConvBlock(512)
        self.conv_4_2 = ConvBlock(512)
        self.conv_4_3 = ConvBlock(512)
        self.dropout_4 = layers.Dropout(self.rate)

        self.conv_5_1 = ConvBlock(512)
        self.conv_5_2 = ConvBlock(512)
        self.conv_5_3 = ConvBlock(512)
        self.dropout_5 = layers.Dropout(self.rate)

        # Decoding

        self.conv_6_1 = ConvBlock(512)
        self.conv_6_2 = ConvBlock(512)
        self.conv_6_3 = ConvBlock(512)
        self.dropout_6 = layers.Dropout(self.rate)

        self.conv_7_1 = ConvBlock(512)
        self.conv_7_2 = ConvBlock(512)
        self.conv_7_3 = ConvBlock(256)
        self.dropout_7 = layers.Dropout(self.rate)

        self.conv_8_1 = ConvBlock(256)
        self.conv_8_2 = ConvBlock(256)
        self.conv_8_3 = ConvBlock(128)
        self.dropout_8 = layers.Dropout(self.rate)

        self.conv_9_1 = ConvBlock(128)
        self.conv_9_2 = ConvBlock(64)

        self.conv_10_1 = ConvBlock(64)

        self.fc = layers.Conv2D(self.num_classes, kernel_size=[1, 1], padding="same")

    def call(self, x, training=False):

        # Encoding

        x = self.conv_1_1(x, training=training)
        x = self.conv_1_2(x, training=training)
        skip_1 = x
        x, index1 = tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        x = self.conv_2_1(x, training=training)
        x = self.conv_2_2(x, training=training)
        skip_2 = x
        x, index2 = tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        x = self.conv_3_1(x, training=training)
        x = self.conv_3_2(x, training=training)
        x = self.conv_3_3(x, training=training)
        skip_3 = x
        x, index3 = tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        if self.rate > 0.0:
            x = self.dropout_3(x, training=True)

        x = self.conv_4_1(x, training=training)
        x = self.conv_4_2(x, training=training)
        x = self.conv_4_3(x, training=training)
        skip_4 = x
        x, index4 = tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        if self.rate > 0.0:
            x = self.dropout_4(x, training=True)

        x = self.conv_5_1(x, training=training)
        x = self.conv_5_2(x, training=training)
        x = self.conv_5_3(x, training=training)
        if self.rate > 0.0:
            x = self.dropout_5(x, training=True)

        # Decoding

        x = self.conv_6_1(x, training=training)
        x = self.conv_6_2(x, training=training)
        x = self.conv_6_3(x, training=training)
        if self.rate > 0.0:
            x = self.dropout_6(x, training=True)

        x = upsampling(x, index4, skip_4.shape)
        x = self.conv_7_1(x, training=training)
        x = self.conv_7_2(x, training=training)
        x = self.conv_7_3(x, training=training)
        if self.rate > 0.0:
            x = self.dropout_7(x, training=True)

        x = upsampling(x, index3, skip_3.shape)
        x = self.conv_8_1(x, training=training)
        x = self.conv_8_2(x, training=training)
        x = self.conv_8_3(x, training=training)
        if self.rate > 0.0:
            x = self.dropout_8(x, training=True)

        x = upsampling(x, index2, skip_2.shape)
        x = self.conv_9_1(x, training=training)
        x = self.conv_9_2(x, training=training)

        x = upsampling(x, index1, skip_1.shape)
        x = self.conv_10_1(x, training=training)

        x = self.fc(x)

        return x


class ConvBlock(layers.Layer):

    def __init__(self, filters, kernel_size=None, name="conv-block", **kwargs):
        super(ConvBlock, self).__init__(name=name, **kwargs)
        if kernel_size is None:
            kernel_size = [3, 3]

        self.filters = filters
        self.kernel_size = kernel_size

        self.conv = layers.Conv2D(self.filters, self.kernel_size, padding="same")
        self.batchnorm = layers.BatchNormalization(axis=-1, fused=True)
        self.relu = layers.ReLU()

    def call(self, x, training=False, **kwargs):
        x = self.conv(x)
        x = self.batchnorm(x, training=training)
        x = self.relu(x)

        return x


def upsampling(x, index, output_shape):
    batch_size = tf.shape(x)[0]
    x = tf.reshape(x, [-1])
    batch_range = tf.reshape(tf.range(batch_size, dtype=index.dtype), [batch_size, 1, 1, 1])
    b = tf.ones_like(index) * batch_range
    b = tf.reshape(b, [-1, 1])
    index = tf.reshape(index, [-1, 1])
    index = tf.concat([b, index], 1)
    x = tf.scatter_nd(index, x, shape=[batch_size, output_shape[1] * output_shape[2] * output_shape[3]])
    x = tf.reshape(x, [tf.shape(x)[0], output_shape[1], output_shape[2], output_shape[3]])
    return x
