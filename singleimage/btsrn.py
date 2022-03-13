from utils import *
import tensorflow as tf
from einops.layers.keras import Rearrange


class PConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters_reduc,
                 filters_recon
                 ):
        super(LRStageBlock, self).__init__()
        self.filters_reduc = filters_reduc
        self.filters_recon = filters_recon

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
        
    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs) + inputs
    

class Upsampling(tf.keras.layers.Layer):
    def __init__(self,
                 scale_rate
                 ):
        super(Upsampling, self).__init__()
        if scale_rate not in [2,3,4]:
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
            self.scaler = tf.keras.layers.UpSampling2D(size=self.scale_rate)

    def call(self, inputs, *args, **kwargs):
        for layer in self.residual:
            inputs = layer(inputs) + self.scaler(inputs)
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

        self.patches = Patches(32)
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
                       self.filters
                       ) for _ in range(self.n_lrstage)
        ])
        self.upsampling = Upsampling(self.scale_rate)
        self.hrstage = tf.keras.Sequential([
            PConvBlock(self.filters_reduction,
                       self.filters
                       ) for _ in range(self.n_hrstage)
        ])
        self.depatch = Rearrange('(b hp wp) p1 p2 c -> b (hp p1) (wp p2) c',
                                 hp=2, wp=2
                                 )

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            reconstruction = self.forward(x)
            loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, reconstruction))
        grads = tape.gradient(loss,
                              self.lrstage.trainable_variables +
                              self.upsampling.trainable_variables +
                              self.hrstage.trainable_variables
                              )
        self.optimizer.apply_gradients(
            zip(grads,
                self.lrstage.trainable_variables +
                self.upsampling.trainable_variables +
                self.hrstage.trainable_variables
                )
        )
        psnr = compute_psnr(reconstruction, y)
        ssim = compute_ssim(reconstruction, y)
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}

    @tf.function
    def test_step(self, data):
        x, y = data
        reconstruction = self.forward(x)
        loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, reconstruction))
        psnr = compute_psnr(reconstruction, y)
        ssim = compute_ssim(reconstruction, y)
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}

    @tf.function
    def forward(self, inputs):
        patches = self.patches(inputs)
        lr_patch = self.lrstage(patches)
        hr_patch = self.hrstage(self.upsampling(lr_patch))
        hr = self.depatch(hr_patch)
        return hr + self.shortcut_biocubic(inputs)

    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)
