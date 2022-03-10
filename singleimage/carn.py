from utils import *
from einops import rearrange
from einops.layers.keras import Rearrange
import tensorflow as tf


class CascadeBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 ):
        super(CascadeBlock, self).__init__()
        self.n_filters = n_filters

        self.residuals = [
            tf.keras.layers.Conv2D(filters=self.n_filters,
                                   kernel_size=3,
                                   strides=1,
                                   padding='SAME',
                                   activation='relu'
                                   ) for _ in range(3)
        ]
        self.pointwises = [
            tf.keras.layers.Conv2D(self.n_filters,
                                   kernel_size=1,
                                   strides=1,
                                   padding='VALID',
                                   activation='linear'
                                   )
        ]

    def call(self, inputs, *args, **kwargs):
        for r, p in zip(self.residuals, self.pointwises):
            residual = r(inputs)
            inputs = p(tf.concat([inputs, residual],
                                 axis=-1
                                 )
                       )
        return inputs


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


class CARN(tf.keras.models.Model):
    def __init__(self,
                 n_filters=64,
                 scale_rate=4
                 ):
        super(CARN, self).__init__()
        self.n_filters = n_filters
        self.scale_rate = scale_rate
        self.rgb_mean = tf.broadcast_to((0.4488, 0.4371, 0.4040),
                                        shape=(1, 1, 1, 3)
                                        )

        self.extractor = tf.keras.Sequential([
            Patches(32),
            tf.keras.layers.Conv2D(self.n_filters,
                                   kernel_size=3,
                                   strides=1,
                                   padding='SAME',
                                   activation='linear'
                                   )
        ])
        self.blocks = [
            CascadeBlock(self.n_filters) for _ in range(3)
        ]
        self.pointwises = [
            tf.keras.layers.Conv2D(self.n_filters,
                                   kernel_size=1,
                                   padding='VALID',
                                   strides=1,
                                   activation='linear'
                                   ) for _ in range(3)
        ]
        self.upsampler = tf.keras.Sequential([
            Upsample(self.n_filters,
                     scale_rate
                     ),
            tf.keras.layers.Conv2D(3,
                                   kernel_size=3,
                                   activation='linear',
                                   padding='SAME',
                                   strides=1
                                   ),
            Rearrange('(b hp wp) p1 p2 c -> b (hp p1) (wp p2) c',
                      hp=2, wp=2
                      )
        ])

    def get_trainable_variables(self):
        blocks = [i.trainable_variables for i in self.blocks]
        points = [i.trainable_variables for i in self.pointwises]
        variables = blocks + points
        return variables[0]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            x = x - self.rgb_mean
            featuremap = self.extractor(x)
            for b, p in zip(self.blocks, self.pointwises):
                blockoutput = b(featuremap)
                featuremap = p(tf.concat([featuremap, blockoutput],
                                         axis=-1
                                         )
                               )
            reconstruction = self.upsampler(featuremap) + self.rgb_mean
            loss = tf.reduce_mean(tf.keras.losses.mean_absolute_error(y, reconstruction))
        grads = tape.gradient(loss,
                              self.extractor.trainable_variables +
                              self.upsampler.trainable_variables +
                              self.get_trainable_variables()
                              )
        self.optimizer.apply_gradients(
            zip(grads,
                self.extractor.trainable_variables +
                self.upsampler.trainable_variables +
                self.get_trainable_variables()
                )
        )
        psnr = compute_psnr(reconstruction, y)
        ssim = compute_ssim(reconstruction, y)
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}

    def test_step(self, data):
        x, y = data
        x = x - self.rgb_mean
        featuremap = self.extractor(x)
        for b, p in zip(self.blocks, self.pointwises):
            blockoutput = b(featuremap)
            featuremap = p(tf.concat([featuremap, blockoutput],
                                     axis=-1
                                     )
                           )
        reconstruction = self.upsampler(featuremap) + self.rgb_mean
        loss = tf.reduce_mean(tf.keras.losses.mean_absolute_error(y, reconstruction))
        psnr = compute_psnr(reconstruction, y)
        ssim = compute_ssim(reconstruction, y)
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}

    def call(self, inputs, training=None, mask=None):
        mu, _ = tf.nn.moments(inputs,
                              axes=[1, 2],
                              keepdims=True
                              )
        inputs = inputs - mu
        feature = self.extractor(inputs)
        for b, p in zip(self.blocks, self.pointwises):
            blockoutput = b(feature)
            feature = p(tf.concat([feature, blockoutput],
                                  axis=-1
                                  )
                        )
        reconstruction = self.upsampler(feature) + mu
        return reconstruction
