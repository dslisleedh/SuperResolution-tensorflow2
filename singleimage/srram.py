from utils import PixelShuffle
import tensorflow as tf
'''
https://deepai.org/publication/ram-residual-attention-module-for-single-image-super-resolution
use L1 loss to reconstruction
'''


class RAM(tf.keras.layers.Layer):
    def __init__(self,
                 c
                 ):
        super(RAM, self).__init__()
        self.c = c

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.c,
                                   kernel_size=3,
                                   strides=1,
                                   padding='SAME',
                                   activation='relu'
                                   ),
            tf.keras.layers.Conv2D(filters=self.c,
                                   kernel_size=3,
                                   strides=1,
                                   padding='SAME',
                                   activation='linear'
                                   )
        ])
        self.ca = tf.keras.Sequential([
            tf.keras.layers.Dense(self.c // 16,
                                  activation='relu'
                                  ),
            tf.keras.layers.Dense(self.c,
                                  activation='linear'
                                  )
        ])
        self.sa = tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                                  padding='SAME',
                                                  activation='linear'
                                                  )

    def call(self, inputs, *args, **kwargs):
        residual = self.forward(inputs)
        _, var = tf.nn.moments(inputs,
                               axes=[1, 2],
                               keepdims=True
                               )
        residual *= tf.nn.sigmoid(self.ca(var) + self.sa(inputs))
        return inputs + residual


class SRRAM(tf.keras.models.Model):
    def __init__(self,
                 upsample_rate,
                 c=64,
                 r=16
                 ):
        super(SRRAM, self).__init__()
        self.c = c
        self.r = r
        self.upsample_rate = upsample_rate

        self.feature_extractor = tf.keras.layers.Conv2D(self.c,
                                                        kernel_size=3,
                                                        padding='SAME',
                                                        strides=1,
                                                        activation='linear'
                                                        )
        self.residual_blocks = tf.keras.Sequential([
            RAM(self.c) for _ in range(self.r)
        ] + [
            tf.keras.layers.Conv2D(self.c,
                                   kernel_size=3,
                                   padding='SAME',
                                   strides=1,
                                   activation='linear'
                                   )
        ])
        if self.upsample_rate == 4:
            self.upscaling = tf.keras.Sequential([
                tf.keras.layers.Conv2D(self.c * (2 ** 2),
                                       kernel_size=3,
                                       activation='relu',
                                       padding='SAME',
                                       strides=1
                                       ),
                PixelShuffle(2),
                tf.keras.layers.Conv2D(self.c * (2 ** 2),
                                       kernel_size=3,
                                       activation='relu',
                                       padding='SAME',
                                       strides=1
                                       ),
                PixelShuffle(2),
                tf.keras.layers.Conv2D(3,
                                       kernel_size=1,
                                       activation='sigmoid',
                                       padding='VALID'
                                       )
            ])
        else:
            self.upscaling = tf.keras.Sequential([
                tf.keras.layers.Conv2D(self.c * (self.upsample_rate ** 2),
                                       kernel_size=3,
                                       activation='relu',
                                       padding='SAME',
                                       strides=1
                                       ),
                PixelShuffle(self.upsample_rate),
                tf.keras.layers.Conv2D(3,
                                       kernel_size=1,
                                       activation='sigmoid',
                                       padding='VALID'
                                       )
            ])

    def forward(self, x, training=False):
        x = self.feature_extractor(x, training=training)
        x = self.residual_blocks(x, training=training) + x
        x = self.upscaling(x, training=training)
        return x

    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)
