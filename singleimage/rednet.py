from utils import *
from einops.layers.keras import Rearrange
import tensorflow as tf


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 downscale=False
                 ):
        super(ConvBlock, self).__init__()
        self.n_filters = n_filters
        self.downscale = downscale

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.n_filters,
                                   kernel_size=3,
                                   strides=2 if self.downscale else 1,
                                   activation='relu',
                                   padding='SAME'
                                   ),
            tf.keras.layers.Conv2D(filters=self.n_filters,
                                   kernel_size=3,
                                   strides=1,
                                   activation='relu',
                                   padding='SAME'
                                   )

        ])

    @tf.function
    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class TransConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 upscale=False
                 ):
        super(TransConvBlock, self).__init__()
        self.n_filters = n_filters
        self.downscale = upscale

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=self.n_filters,
                                            kernel_size=3,
                                            strides=1,
                                            activation='relu',
                                            padding='SAME'
                                            ),
            tf.keras.layers.Conv2DTranspose(filters=self.n_filters,
                                            kernel_size=3,
                                            strides=2 if self.downscale else 1,
                                            activation='relu',
                                            padding='SAME'
                                            )

        ])

    @tf.function
    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class REDNet(tf.keras.models.Model):
    def __init__(self,
                 n_filters,
                 n_layers
                 ):
        super(REDNet, self).__init__()
        self.n_filters = n_filters
        self.n_layers = n_layers

        self.encoder = [
            ConvBlock(self.n_filters,
                      downscale=True if i == 0 else 1
                      ) for i in range(self.n_layers//2)
        ]
        self.decoder = [
            TransConvBlock(self.n_filters,
                           upscale=True if i == (self.n_layers//2)-1 else False
                           ) for i in range(self.n_layers//2)
        ]

    def return_variable(self):
        encoder_trainable_variable = [layer.trainable_variables for layer in self.encoder]
        decoder_trainable_variable = [layer.trainable_variables for layer in self.decoder]
        total_variable = encoder_trainable_variable + decoder_trainable_variable
        return total_variable[0]

    @tf.function
    def forward(self, x):
        skip = [x]
        for layer in self.encoder:
            x = layer(x)
            skip.append(x)
        for layer, s in zip(self.decoder, skip):
            x = layer(x)
            x = tf.nn.relu(x + s)
        return x

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            reconstruction = self.forward(x)
            loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, reconstruction))
        grads = tape.gradient(loss, self.return_variable())
        self.optimizer.apply_gradients(
            zip(grads, self.return_variable())
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
        return {'loss': loss, 'pnsr': psnr, 'ssim': ssim}

    @tf.function
    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)