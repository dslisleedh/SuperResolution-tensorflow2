from utils import *
from einops.layers.keras import Rearrange
import tensorflow as tf


'''
https://www4.comp.polyu.edu.hk/~cslzhang/paper/IRCNN_CVPR17.pdf
used batch normalization to train to accelerate train speed
samples with small size to help avoid boundary artifacts : split input into patches
used dilated convolution to enlarge receptive field

*** learn residual image y - x
'''
class DConv(tf.keras.layers.Layer):
    def __init__(self,
                 d,
                 n_filters,
                 bn
                 ):
        super(DConv, self).__init__()
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
                 scale_rate=4,
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

    @tf.function
    def train_step(self, data):
        x, y = data
        x = self.upscaler(x)
        with tf.GradientTape() as tape:
            pred = self.forward(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, pred))
        grads = tape.gradient(loss, self.forward.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.forward.trainable_variables)
        )
        psnr, ssim = compute_metrics(pred, y)
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}

    @tf.function
    def test_step(self, data):
        x, y = data
        x = self.upscaler(x)
        pred = self.forward(x)
        loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, pred))
        psnr, ssim = compute_metrics(pred, y)
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}

    @tf.function
    def call(self, inputs, training=None, mask=None):
        inputs = self.upscaler(inputs)
        residual = self.forward(inputs)
        return self.DePatch(residual) + self.DePatch(inputs)

