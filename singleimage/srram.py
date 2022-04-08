from utils import PixelShuffle
import tensorflow as tf



class RAM(tf.keras.layers.Layer):
    def __init__(self,
                 c,
                 r
                 ):
        super(RAM, self).__init__()
        self.c = c
        self.r = r

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
            tf.keras.layers.Dense(self.r,
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
                 r,
                 upsample_rate,
                 c=64,
                 n_blocks=16
                 ):
        super(SRRAM, self).__init__()
        self.c = c
        self.r = r
        self.n_blocks = n_blocks
        self.upsample_rate = upsample_rate

        self.feature_extractor = tf.keras.layers.Conv2D(self.c,
                                                        kernel_size=3,
                                                        padding='SAME',
                                                        strides=1,
                                                        activation='linear'
                                                        )
        self.residual_blocks = tf.keras.Sequential([
            RAM(self.c, self.r) for _ in range(self.n_blocks)
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

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.residual_blocks(x) + x
        x = self.upscaling(x)
        return x

    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)
