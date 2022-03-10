from utils import *
from einops import rearrange
from einops.layers.keras import Rearrange
import tensorflow as tf


'''
https://arxiv.org/abs/1609.05158
used pixel_shuffle to upsample + patch_learning
'''
class ESPConvLayer(tf.keras.layers.Layer):
    def __init__(self,
                 num_filters,
                 scale_rate
                 ):
        super(ESPConvLayer, self).__init__()
        self.num_filters = num_filters
        self.scale_rate = scale_rate


        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.num_filters * (self.scale_rate ** 2),
                                   kernel_size=3,
                                   padding='SAME',
                                   activation='linear'
                                   )
        ])

    def call(self, inputs, *args, **kwargs):
        inputs = tf.nn.depth_to_space(self.forward(inputs),
                                      block_size=self.scale_rate
                                      )
        return tf.nn.sigmoid(inputs)


class ESPCN(tf.keras.models.Model):
    def __init__(self,
                 n_filters=[64, 32],
                 kernel_size=[5, 3],
                 scale_rate=4
                 ):
        super(ESPCN, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.scale_rate = scale_rate

        self.forward = tf.keras.Sequential([Patches(16)])
        for f, k in zip(self.n_filters, self.kernel_size):
            self.forward.add(tf.keras.layers.Conv2D(f,
                                                    kernel_size=k,
                                                    strides=1,
                                                    padding='SAME',
                                                    activation='tanh'
                                                    )
                             )
        self.forward.add(ESPConvLayer(3, self.scale_rate))
        self.forward.add(Rearrange('(b hp wp) p1 p2 c -> b (hp p1) (wp p2) c',
                                   hp=64//16, wp=64//16
                                   )
                         )

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            pred = self.forward(x)
            loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, pred))
        grads = tape.gradient(loss, self.forward.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.forward.trainable_variables)
        )
        psnr = compute_psnr(pred, y)
        ssim = compute_ssim(pred, y)
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}

    @tf.function
    def test_step(self, data):
        x, y = data
        pred = self.forward(x)
        loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, pred))
        psnr = compute_psnr(pred, y)
        ssim = compute_ssim(pred, y)
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}

    @tf.function
    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)


