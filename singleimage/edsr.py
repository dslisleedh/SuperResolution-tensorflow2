from utils import *
from einops import rearrange
from einops.layers.keras import Rearrange
import tensorflow as tf


'''
Used residual architecture and L1 loss
Removed batch normalization and post addition ReLU
used pixel shuffle to Upscale
'''
class ResBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 scaling_factor=.1
                 ):
        super(ResBlock, self).__init__()
        self.n_filters = n_filters
        self.scaling_factor = scaling_factor

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.n_filters,
                                   kernel_size=3,
                                   padding='SAME',
                                   strides=1,
                                   activation='relu'
                                   ),
            tf.keras.layers.Conv2D(self.n_filters,
                                   kernel_size=3,
                                   padding='SAME',
                                   strides=1,
                                   activation='linear'
                                   )
        ])

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs) * self.scaling_factor + inputs


class Upsample(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 scale_ratio
                 ):
        super(Upsample, self).__init__()
        self.n_filters = n_filters
        if scale_ratio not in [2, 3, 4]:
            raise ValueError('NotImplemented')
        else:
            self.scale_ratio = scale_ratio

        if self.scale_ratio == 4:
            self.forward = tf.keras.Sequential([
                tf.keras.layers.Conv2D(n_filters * (2 ** 2),
                                       kernel_size=3,
                                       strides=1,
                                       padding='SAME',
                                       activation='relu'
                                       ),
                PixelShuffle(2),
                tf.keras.layers.Conv2D(n_filters * (2 ** 2),
                                       kernel_size=3,
                                       strides=1,
                                       padding='SAME',
                                       activation='relu'
                                       ),
                PixelShuffle(2)
            ])
        else:
            self.forward = tf.keras.Sequential([
                tf.keras.layers.Conv2D(n_filters * (self.scale_ratio ** 2),
                                       kernel_size=3,
                                       strides=1,
                                       padding='SAME',
                                       activation='relu'
                                       ),
                PixelShuffle(self.scale_ratio)
            ])

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class EDSR(tf.keras.models.Model):
    def __init__(self,
                 n_filters=32,
                 n_blocks=256,
                 scale_rate=4
                 ):
        super(EDSR, self).__init__()
        self.n_filters = n_filters
        self.n_blocks = n_blocks
        self.scale_rate = scale_rate
        ### In paper, they normalized dataset(DIV2K)
        self.rgb_mean = tf.broadcast_to((0.4488, 0.4371, 0.4040),
                                        shape=(1, 1, 1, 3)
                                        )

        self.extractor = tf.keras.layers.Conv2D(self.n_filters,
                                                kernel_size=3,
                                                activation='relu',
                                                strides=1,
                                                padding='SAME'
                                                )
        self.resblocks = tf.keras.Sequential([
            ResBlock(self.n_filters) for _ in range(self.n_blocks)
        ] + [
            tf.keras.layers.Conv2D(self.n_filters,
                                   activation='linear',
                                   strides=1,
                                   kernel_size=3,
                                   padding='SAME'
                                   )
        ])
        self.upsampler = tf.keras.Sequential([
            Upsample(self.n_filters,
                     scale_ratio=self.scale_rate
                     ),
            tf.keras.layers.Conv2D(3,
                                   kernel_size=3,
                                   padding='SAME',
                                   strides=1,
                                   activation='linear'
                                   )
        ])

    @tf.function
    def train_step(self, data):
        x, y = data
        x = self.get_patches(x) - self.rgb_mean
        with tf.GradientTape() as tape:
            featuremap = self.extractor(x, training=True)
            featuremap = self.resblocks(featuremap, training=True) + featuremap
            reconstruction = self.upsampler(featuremap, training=True) + self.rgb_mean
            loss = tf.reduce_mean(tf.keras.losses.mean_absolute_error(y, reconstruction))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables)
        )
        psnr, ssim = compute_metrics(reconstruction, y)
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}

    @tf.function
    def test_step(self, data):
        x, y = data
        x = x - self.rgb_mean
        featuremap = self.extractor(x)
        featuremap = self.resblocks(featuremap) + featuremap
        reconstruction = self.upsampler(featuremap) + self.rgb_mean
        loss = tf.reduce_mean(tf.keras.losses.mean_absolute_error(y, reconstruction))
        psnr, ssim = compute_metrics(reconstruction, y)
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}

    @tf.function
    def call(self, inputs, training=None, mask=None):
        mu, _ = tf.nn.moments(inputs,
                              axes=[1,2],
                              keepdims=True
                              )
        featuremap = self.extractor(inputs)
        featuremap = self.resblocks(featuremap) + featuremap
        return self.upsampler(featuremap) + mu
