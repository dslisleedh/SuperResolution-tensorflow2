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
        self.upscale = upscale

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=self.n_filters,
                                            kernel_size=3,
                                            strides=1,
                                            activation='relu',
                                            padding='SAME'
                                            ),
            tf.keras.layers.Conv2DTranspose(filters=3 if self.upscale else self.n_filters,
                                            kernel_size=3,
                                            strides=2 if self.upscale else 1,
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
                 n_layers,
                 scale_rate=4
                 ):
        super(REDNet, self).__init__()
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.scale_rate = scale_rate

        self.upscale = BicubicScale2D(self.scale_rate)
        self.encoder = [
            ConvBlock(self.n_filters,
                      downscale=True if i == 0 else False
                      ) for i in range(self.n_layers // 2)
        ]
        self.decoder = [
            TransConvBlock(self.n_filters,
                           upscale=True if i == (self.n_layers // 2) - 1 else False
                           ) for i in range(self.n_layers // 2)
        ]

    @tf.function
    def forward(self, x, training=False):
        mu, _ = tf.nn.moments(x,
                              axes=[1,2],
                              keepdims=True
                              )
        x = x - mu
        x = self.upscale(x, training=training)
        skip = [x]
        for idx, layer in enumerate(self.encoder):
            x = layer(x, training=training)
            if idx != (self.n_layers // 2) - 1:
                skip.append(x)
        skip = skip[::-1]
        for layer, s in zip(self.decoder, skip):
            x = layer(x, training=training)
            x = tf.nn.relu(x + s)
        x = x + mu
        return x

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
        return {'loss': loss, 'pnsr': psnr, 'ssim': ssim}

    @tf.function
    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)
