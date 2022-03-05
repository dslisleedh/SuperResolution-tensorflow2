import tensorflow as tf
from utils import BiocubicUpsampling2D


'''
https://arxiv.org/abs/1501.00092
Use MSE Loss to train
'''
class SRCNN(tf.keras.models.Model):
    def __init__(self,
                 expansion_rate=4
                 ):
        super(SRCNN, self).__init__()
        self.expansion_rate = expansion_rate

        self.forward = tf.keras.Sequential([
            BiocubicUpsampling2D(self.expansion_rate),
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=9,
                                   padding='SAME',
                                   strides=1,
                                   activation='relu'
                                   ),
            tf.keras.layers.Conv2D(filters=32,
                                   kernel_size=1,
                                   padding='VALID',
                                   strides=1,
                                   activation='relu'
                                   ),
            tf.keras.layers.Conv2D(filters=3,
                                   kernel_size=5,
                                   padding='SAME',
                                   strdies=1,
                                   activation='linear'
                                   )
        ])

    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)
