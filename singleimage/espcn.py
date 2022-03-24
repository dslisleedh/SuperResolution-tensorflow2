from utils import *
from einops import rearrange
from einops.layers.keras import Rearrange
import tensorflow as tf


'''
https://arxiv.org/abs/1609.05158
used pixel_shuffle to upsample + patch_learning
'''


class RecursiveBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 u
                 ):
        super(RecursiveBlock, self).__init__()
        self.n_filters = n_filters
        self.u = u

        self.forward = tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(self.n_filters,
                                   kernel_size=3,
                                   activation='linear',
                                   kernel_initializer='he_normal',
                                   strides=1,
                                   padding='SAME',
                                   use_bias=False
                                   ),
            tf.keras.layers.ReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(self.n_filters,
                                   kernel_size=3,
                                   activation='linear',
                                   kernel_initializer='he_normal',
                                   strides=1,
                                   padding='SAME',
                                   use_bias=False
                                   )
        ])

    def call(self, inputs, *args, **kwargs):
        residual = self.forward(inputs)
        for _ in range(self.u):
            residual = self.forward(inputs + residual)
        return inputs + residual


class DRRN(tf.keras.models.Model):
    def __init__(self,
                 n_filters=128,
                 b=1,
                 u=25,
                 theta=.01,
                 scale_rate=4
                 ):
        super(DRRN, self).__init__()
        self.n_filters = n_filters
        self.b = b
        self.u = u
        self.theta = theta
        self.scale_rate = scale_rate

        self.upscaler = BicubicScale2D(self.scale_rate)
        self.feature_extractor = tf.keras.layers.Conv2D(self.n_filters,
                                                        kernel_size=3,
                                                        activation='linear',
                                                        strides=1,
                                                        padding='SAME'
                                                        )
        self.recursive_blocks = tf.keras.Sequential([
            RecursiveBlock(self.n_filters,
                           self.u
                           ) for _ in range(self.b)
        ])
        self.reconstruction_network = tf.keras.layers.Conv2D(3,
                                                             kernel_size=3,
                                                             activation='linear',
                                                             strides=1,
                                                             padding='SAME'
                                                             )

    @tf.function
    def forward(self, x, training=False):
        x = self.upscaler(x)
        featuremap = self.feature_extractor(x, training=training)
        featuremap = self.recursive_blocks(featuremap, training=training)
        x = x + self.reconstruction_network(featuremap, training=training)
        return x

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            reconstruction = self.forward(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, reconstruction))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip([tf.clip_by_value(g,
                                  -self.theta / self.optimizer.learning_rate,
                                  self.theta / self.optimizer.learning_rate
                                  ) for g in grads
                 ],
                self.trainable_variables
                )
        )
        psnr, ssim = compute_metrics(y, reconstruction)
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}

    @tf.function
    def test_step(self, data):
        x, y = data
        reconstruction = self.forward(x)
        loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, reconstruction))
        psnr, ssim = compute_metrics(y, reconstruction)
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}

    @tf.function
    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)


