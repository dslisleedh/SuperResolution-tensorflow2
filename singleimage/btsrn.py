from utils import *
import tensorflow as tf
from einops.layers.keras import Rearrange


class PConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters_reduc,
                 filters_recon,
                 skip_channel=False
                 ):
        super(PConvBlock, self).__init__()
        self.filters_reduc = filters_reduc
        self.filters_recon = filters_recon
        self.skip_channel = skip_channel

        self.forward = tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(self.filters_reduc,
                                   kernel_size=1,
                                   padding='VALID',
                                   activation='linear',
                                   strides=1
                                   ),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(self.filters_recon,
                                   kernel_size=3,
                                   padding='SAME',
                                   activation='linear',
                                   strides=1
                                   )
        ])
        if self.skip_channel:
            self.skip = tf.keras.layers.Conv2D(self.filters_recon,
                                               kernel_size=1,
                                               padding='VALID',
                                               strides=1,
                                               activation='linear'
                                               )
        else:
            self.skip = tf.keras.layers.Layer()

    @tf.function
    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs) + self.skip(inputs)


class Upsampling(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 scale_rate
                 ):
        super(Upsampling, self).__init__()
        self.n_filters = n_filters
        if scale_rate not in [2, 3, 4]:
            raise ValueError('Unsupported scale rate')
        else:
            self.scale_rate = scale_rate

        if self.scale_rate == 4:
            self.residual = [
                tf.keras.layers.Conv2DTranspose(3,
                                                kernel_size=3,
                                                activation='linear',
                                                strides=2,
                                                padding='SAME'
                                                ) for _ in range(2)
            ]
            self.upscaler = tf.keras.layers.UpSampling2D(size=2)
        else:
            self.residual = [
                tf.keras.layers.Conv2DTranspose(3,
                                                kernel_size=3,
                                                activation='linear',
                                                strides=self.scale_rate,
                                                padding='SAME'
                                                )
            ]
            self.upscaler = tf.keras.layers.UpSampling2D(size=self.scale_rate)

    def call(self, inputs, *args, **kwargs):
        for layer in self.residual:
            inputs = layer(inputs) + self.upscaler(inputs)
        return inputs


class BTSRN(tf.keras.models.Model):
    def __init__(self,
                 scale_rate,
                 filters,
                 filters_reduction,
                 n_lrstage=6,
                 n_hrstage=4
                 ):
        super(BTSRN, self).__init__()
        self.scale_rate = scale_rate
        self.filters = filters
        self.filters_reduction = filters_reduction
        self.n_lrstage = n_lrstage
        self.n_hrstage = n_hrstage

        self.shortcut_biocubic = BicubicScale2D(rate=self.scale_rate)
        self.lrstage = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.filters,
                                   kernel_size=1,
                                   activation='linear',
                                   padding='VALID',
                                   strides=1
                                   )
        ] + [
            PConvBlock(self.filters_reduction,
                       self.filters if i != self.n_lrstage - 1 else 3,
                       skip_channel=False if i != self.n_lrstage - 1 else 3
                       ) for i in range(self.n_lrstage)
        ])
        self.upsampling = Upsampling(self.filters,
                                     self.scale_rate
                                     )
        self.hrstage = tf.keras.Sequential([
            PConvBlock(self.filters_reduction,
                       self.filters if i != self.n_hrstage - 1 else 3,
                       skip_channel=True if i == 0 else i if i == self.n_hrstage - 1 else False
                       ) for i in range(self.n_hrstage)
        ])

    @tf.function
    def forward(self, inputs, training=False):
        lr = self.lrstage(inputs, training=training)
        hr = self.hrstage(self.upsampling(lr), training=training)
        return hr + self.shortcut_biocubic(inputs)

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            reconstruction = self.forward(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, reconstruction))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables)
        )
        psnr, ssim = compute_metrics(reconstruction, y)
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}

    @tf.function
    def test_step(self, data):
        x, y = data
        reconstruction = self.forward(x)
        loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, reconstruction))
        psnr, ssim = compute_metrics(reconstruction, y)
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}

    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)
