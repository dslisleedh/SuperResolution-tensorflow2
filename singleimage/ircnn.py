import tensorflow as tf
from utils import BicubicScale2D


class DConv(tf.keras.layers.Layer):
    def __init__(self,
                 d,
                 n_filters,
                 bn
                 ):
        self.d = d
        self.n_filters = n_filters
        self.bn = bn

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.n_filters,
                                   strides=1,
                                   padding='SAME',
                                   kernel_size=3,
                                   dilation_rate=self.d,
                                   activation='linear',
                                   use_bias=False if self.bn else True
                                   )
        ])
        if self.bn:
            self.forward.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, *args, **kwargs):
        return tf.nn.relu(self.forward(inputs))


class IRCNN(tf.keras.models.Model):
    def __init__(self,
                 scale_rate=3,
                 d=[1, 2, 3, 4, 3, 2, 1],
                 output_channel=3
                 ):
        super(IRCNN, self).__init__()
        self.scale_rate = scale_rate
        self.d = d
        self.output_channels = output_channel

        self.upscaler = BicubicScale2D(self.scale_rate)
        self.forward = tf.keras.Sequential([
            DConv(d=dil,
                  n_filters=64 if i != len(d) - 1 else 3,
                  bn=False if i == 0 else True
                  ) for i, dil in enumerate(d)
        ])

    def call(self, inputs, training=None, mask=None):
        inputs = self.upscaler(inputs)
        return self.forward(inputs) + inputs

