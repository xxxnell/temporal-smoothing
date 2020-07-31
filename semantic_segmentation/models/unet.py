import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model


class UNet(Model):

    def __init__(self, num_classes, rate=0.0, name="u-net"):
        super(UNet, self).__init__(name=name)
        self.num_classes = num_classes
        self.rate = rate
        self.strides = [2, 2]

        # Encoding

        self.conv_1_1 = ConvBlock(64)
        self.conv_1_2 = ConvBlock(64)

        self.pool_2 = layers.MaxPool2D(pool_size=self.strides, strides=self.strides, padding="same")
        self.conv_2_1 = ConvBlock(128)
        self.conv_2_2 = ConvBlock(128)

        self.pool_3 = layers.MaxPool2D(pool_size=self.strides, strides=self.strides, padding="same")
        self.conv_3_1 = ConvBlock(256)
        self.conv_3_2 = ConvBlock(256)
        self.conv_3_3 = ConvBlock(256)
        self.dropout_3 = layers.Dropout(self.rate)

        self.pool_4 = layers.MaxPool2D(pool_size=self.strides, strides=self.strides, padding="same")
        self.conv_4_1 = ConvBlock(512)
        self.conv_4_2 = ConvBlock(512)
        self.conv_4_3 = ConvBlock(512)
        self.dropout_4 = layers.Dropout(self.rate)

        self.pool_5 = layers.MaxPool2D(pool_size=self.strides, strides=self.strides, padding="same")
        self.conv_5_1 = ConvBlock(1024)
        self.conv_5_2 = ConvBlock(1024)
        self.conv_5_3 = ConvBlock(1024)
        self.dropout_5 = layers.Dropout(self.rate)

        # Decoding

        self.deconv_6 = DeconvBlock(1024, 512, strides=self.strides)
        self.conv_6_1 = ConvBlock(512)
        self.conv_6_2 = ConvBlock(512)
        self.conv_6_3 = ConvBlock(512)
        self.dropout_6 = layers.Dropout(self.rate)

        self.deconv_7 = DeconvBlock(512, 256, strides=self.strides)
        self.conv_7_1 = ConvBlock(256)
        self.conv_7_2 = ConvBlock(256)
        self.conv_7_3 = ConvBlock(256)
        self.dropout_7 = layers.Dropout(self.rate)

        self.deconv_8 = DeconvBlock(256, 128, strides=self.strides)
        self.conv_8_1 = ConvBlock(128)
        self.conv_8_2 = ConvBlock(128)
        self.conv_8_3 = ConvBlock(128)
        self.dropout_8 = layers.Dropout(self.rate)

        self.deconv_9 = DeconvBlock(128, 64, strides=self.strides)
        self.conv_9_1 = ConvBlock(64)
        self.conv_9_2 = ConvBlock(64)

        self.fc = layers.Conv2D(self.num_classes, kernel_size=[1, 1], padding="same")

    def call(self, x, training=False):
        # Encoding

        x = self.conv_1_1(x, training=training)
        x = self.conv_1_2(x, training=training)
        skip_1 = x

        x = self.pool_2(x)
        x = self.conv_2_1(x, training=training)
        x = self.conv_2_2(x, training=training)
        skip_2 = x

        x = self.pool_3(x)
        x = self.conv_3_1(x, training=training)
        x = self.conv_3_2(x, training=training)
        x = self.conv_3_3(x, training=training)
        skip_3 = x
        if self.rate > 0.0:
            x = self.dropout_3(x, training=True)

        x = self.pool_4(x)
        x = self.conv_4_1(x, training=training)
        x = self.conv_4_2(x, training=training)
        x = self.conv_4_3(x, training=training)
        skip_4 = x
        if self.rate > 0.0:
            x = self.dropout_4(x, training=True)

        x = self.pool_5(x)
        x = self.conv_5_1(x, training=training)
        x = self.conv_5_2(x, training=training)
        x = self.conv_5_3(x, training=training)
        if self.rate > 0.0:
            x = self.dropout_5(x, training=True)

        # Decoding

        x = self.deconv_6(x, skip_4.shape, training=training)
        x = tf.add(x, skip_4)
        x = self.conv_6_1(x, training=training)
        x = self.conv_6_2(x, training=training)
        x = self.conv_6_3(x, training=training)
        if self.rate > 0.0:
            x = self.dropout_6(x, training=True)

        x = self.deconv_7(x, skip_3.shape, training=training)
        x = tf.add(x, skip_3)
        x = self.conv_7_1(x, training=training)
        x = self.conv_7_2(x, training=training)
        x = self.conv_7_3(x, training=training)
        if self.rate > 0.0:
            x = self.dropout_7(x, training=True)

        x = self.deconv_8(x, skip_2.shape, training=training)
        x = tf.add(x, skip_2)
        x = self.conv_8_1(x, training=training)
        x = self.conv_8_2(x, training=training)
        x = self.conv_8_3(x, training=training)
        if self.rate > 0.0:
            x = self.dropout_8(x, training=True)

        x = self.deconv_9(x, skip_1.shape, training=training)
        x = tf.add(x, skip_1)
        x = self.conv_9_1(x, training=training)
        x = self.conv_9_2(x, training=training)

        x = self.fc(x)

        return x


class ConvBlock(layers.Layer):

    def __init__(self, filters, kernel_size=None):
        super(ConvBlock, self).__init__()
        if kernel_size is None:
            kernel_size = [3, 3]

        self.filters = filters
        self.kernel_size = kernel_size

        self.conv = layers.Conv2D(self.filters, self.kernel_size, padding="same")
        self.bn = layers.BatchNormalization(axis=-1, fused=True)
        self.relu = layers.ReLU()

    def call(self, x, training=False, **kwargs):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x, training=training)

        return x


class DeconvBlock(layers.Layer):

    def __init__(self, in_filters, out_filters, kernel_size=None, strides=None):
        super(DeconvBlock, self).__init__()
        if kernel_size is None:
            kernel_size = [3, 3]
        if strides is None:
            strides = [2, 2]

        self.kernel_size = kernel_size
        self.strides = strides

        initializer = tf.keras.initializers.GlorotUniform()
        self.filters = tf.Variable(
            initializer(shape=self.kernel_size + [out_filters, in_filters], dtype=tf.float32))
        self.bn = layers.BatchNormalization(axis=-1, fused=True)
        self.relu = layers.ReLU()

    def call(self, x, output_shape, training=False, **kwargs):
        x = tf.nn.conv2d_transpose(x, self.filters, output_shape, self.strides, padding='SAME')
        x = self.relu(x)
        x = self.bn(x, training=training)

        return x

