from utils import PixelShuffle, compute_metrics
import tensorflow as tf
'''
https://arxiv.org/pdf/1811.12043.pdf
use L1 loss to reconstruction
48x48 random cropped patch learning
'''


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
                                   activation='relu',
                                   kernel_initializer='he_normal'
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
                                  activation='relu',
                                  kernel_initializer='he_normal'
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
                 c=64,
                 rgb_mean=[0.4488, 0.4371, 0.4040]
                 ):
        super(MAMNet, self).__init__()
        self.c = c
        self.r = r
        self.upsample_rate = upsample_rate
        self.rgb_mean = tf.broadcast_to(rgb_mean, (1, 1, 1, 3))

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
                                       strides=1,
                                       kernel_initializer='he_normal'
                                       ),
                PixelShuffle(2),
                tf.keras.layers.Conv2D(self.c * (2 ** 2),
                                       kernel_size=3,
                                       activation='relu',
                                       padding='SAME',
                                       strides=1,
                                       kernel_initializer='he_normal'
                                       ),
                PixelShuffle(2)
            ])
        else:
            self.upscaling = tf.keras.Sequential([
                tf.keras.layers.Conv2D(self.c * (self.upsample_rate ** 2),
                                       kernel_size=3,
                                       activation='relu',
                                       padding='SAME',
                                       strides=1,
                                       kernel_initializer='he_normal'
                                       ),
                PixelShuffle(self.upsample_rate)
            ])
        self.upscaling.add(tf.keras.layers.Conv2D(3,
                                                  kernel_size=1,
                                                  strides=1,
                                                  activation='linear',
                                                  padding='VALID'
                                                  )
                           )

    @tf.function
    def forward(self, x, training=False):
        if training:
            x = x - self.rgb_mean
        else:
            mean, _ = tf.nn.moments(x,
                                    axes=[1, 2],
                                    keepdims=True
                                    )
            x = x - mean
        x = self.feature_extractor(x, training=training)
        x = self.residual_blocks(x, training=training) + x
        x = self.upscaling(x, training=training)
        if training:
            x = x + self.rgb_mean
        else:
            x = x + mean
        return x

    def train_step(self, data):
        lr, hr = data
        with tf.GradientTape() as tape:
            reconstruction = self.forward(lr, training=True)
            loss = tf.reduce_mean(
                tf.losses.mean_absolute_error(hr, reconstruction)
            )
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables)
        )
        psnr, ssim = compute_metrics(hr, reconstruction)
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}

    def test_step(self, data):
        lr, hr = data
        reconstruction = self.forward(lr)
        loss = tf.reduce_mean(
            tf.losses.mean_absolute_error(hr, reconstruction)
        )
        psnr, ssim = compute_metrics(hr, reconstruction)
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}

    @tf.function
    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)