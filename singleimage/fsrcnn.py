import tensorflow as tf


'''
https://arxiv.org/abs/1608.00367
use MSE Loss to train
best trade-off parameter in paper : (d = 56, s = 12, m = 4)
'''
class FSRCNN(tf.keras.models.Model):
    def __init__(self,
                 d,
                 s,
                 m,
                 expansion_rate=4,
                 output_channels=3
                 ):
        super(FSRCNN, self).__init__()
        self.d = d
        self.s = s
        self.m = m
        self.expansion_rate = expansion_rate
        self.output_channels = output_channels

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.d,
                                   kernel_size=9,
                                   padding='SAME',
                                   strides=1,
                                   activation=tf.keras.layers.PReLU()
                                   ),
            tf.keras.layers.Conv2D(filters=self.s,
                                   kernel_size=1,
                                   padding='VALID',
                                   strides=1,
                                   activation=tf.keras.layers.PReLU()
                                   )
        ] + [
            tf.keras.layers.Conv2D(filters=self.s,
                                   kernel_size=3,
                                   padding='SAME',
                                   strides=1,
                                   activation='linear'
                                   ) for _ in range(self.m)
        ] + [
            tf.keras.layers.PReLU(),
            tf.keras.layers.Conv2D(filters=self.d,
                                   kernel_size=1,
                                   padding='VALID',
                                   strides=1,
                                   activation=tf.keras.layers.PReLU()
                                   ),
            tf.keras.layers.Conv2DTranspose(filters=self.output_channels,
                                            kernel_size=9,
                                            strides=self.expansion_rate,
                                            padding='SAME',
                                            activation='linear'
                                            )
        ])

    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)
