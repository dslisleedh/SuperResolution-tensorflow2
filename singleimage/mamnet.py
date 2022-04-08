from utils import PixelShuffle
import tensorflow as tf


class MAMB(tf.keras.layers.Layer):
    def __init__(self,
                 c
                 ):
        super(MAMB, self).__init__()
        self.c = c

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.c,
                                   kernel_size=3,
                                   padding='SAME',
                                   strides=1,
                                   activation='relu'
                                   ),
            tf.keras.layers.Conv2D(self.c,
                                   kernel_size=3,
                                   padding='SAME',
                                   strides=1,
                                   activation='linear'
                                   )
        ])
        self.icd = tf.keras.Sequential([
            tf.keras.layers.Dense(self.c//16,
                                  activation='relu'
                                  ),
            tf.keras.layers.Dense(self.c,
                                  activation='linear'
                                  )
        ])
        self.csd = tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                                   activation='linear',
                                                   padding='SAME',
                                                   strides=1
                                                   )

    def call(self, inputs, *args, **kwargs):
        residual = self.forward(inputs)
        _, var = tf.nn.moments(residual,
                               axes=[1, 2],
                               keepdims=True
                               )
        residual *= tf.nn.sigmoid(var + self.icd(var) + self.csd(residual))
        return residual + inputs


class MAMNet(tf.keras.models.Model):
    def __init__(self,
                 upsample_rate,
                 r=16,
                 c=64
                 ):
        super(MAMNet, self).__init__()
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
            MAMB(self.c) for _ in range(self.r)
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