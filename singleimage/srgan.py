from utils import *
import tensorflow as tf


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=3,
                                   strides=1,
                                   padding='SAME',
                                   use_bias=False,
                                   activation='linear'
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=3,
                                   strides=1,
                                   padding='SAME',
                                   use_bias=False,
                                   activation='linear'
                                   ),
            tf.keras.layers.BatchNormalization()
        ])

    @tf.function
    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs) + inputs


class Generator(tf.keras.layers.Layer):
    def __init__(self):
        super(Generator, self).__init__()

        self.input_projection = tf.keras.layers.Conv2D(filters=64,
                                                       kernel_size=9,
                                                       strides=1,
                                                       padding='SAME',
                                                       activation='relu'
                                                       )
        self.residual_blocks = tf.keras.Sequential([
            ResidualBlock() for _ in range(16)
        ] + [
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=3,
                                   strides=1,
                                   padding='SAME',
                                   use_bias=False
                                   ),
            tf.keras.layers.BatchNormalization()
        ])
        self.upsampler = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64 * (2 ** 2),
                                   kernel_size=3,
                                   strides=1,
                                   padding='SAME'
                                   ),
            PixelShuffle(2),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=64 * (2 ** 2),
                                   kernel_size=3,
                                   strides=1,
                                   padding='SAME'
                                   ),
            PixelShuffle(2),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=3,
                                   kernel_size=9,
                                   strides=1,
                                   padding='SAME',
                                   activation=tf.keras.layers.Activation('tanh')
                                   )
        ])

    @tf.function
    def call(self, inputs, *args, **kwargs):
        featuremap = self.input_projection(inputs)
        featuremap = self.residual_blocks(featuremap) + featuremap
        reconstruction = self.upsampler(featuremap)
        return reconstruction


class DiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 strides
                 ):
        super(DiscriminatorBlock, self).__init__()
        self.filters = filters
        self.strides = strides

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.filters,
                                   kernel_size=3,
                                   strides=self.strides,
                                   padding='SAME',
                                   activation='linear',
                                   use_bias=False
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=.2)
        ])

    @tf.function
    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class Discriminator(tf.keras.layers.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=3,
                                   padding='SAME',
                                   strides=1,
                                   activation=tf.keras.layers.LeakyReLU(alpha=.2)
                                   ),
            DiscriminatorBlock(64, 2),
            DiscriminatorBlock(128, 1),
            DiscriminatorBlock(128, 2),
            DiscriminatorBlock(256, 1),
            DiscriminatorBlock(256, 2),
            DiscriminatorBlock(512, 1),
            DiscriminatorBlock(512, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024,
                                  activation=tf.keras.layers.LeakyReLU(alpha=.2)
                                  ),
            tf.keras.layers.Dense(1,
                                  activation='sigmoid'
                                  )
        ])

    @tf.function
    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class SRGAN(tf.keras.models.Model):
    def __init__(self):
        super(SRGAN, self).__init__()

        self.generator = Generator()
        self.discriminator = Discriminator()

        vgg = tf.keras.applications.vgg19.VGG19(include_top=False)
        self.vgg = tf.keras.Model(inputs=vgg.input,
                                  outputs=vgg.layers[5].output
                                  # for vgg5,4 use vgg.layers[-2].output
                                  )
        self.vgg.trainable = False

    def compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                steps_per_execution=None,
                **kwargs):
        super(SRGAN, self).compile(optimizer,
                                   loss,
                                   metrics,
                                   loss_weights,
                                   weighted_metrics,
                                   run_eagerly,
                                   steps_per_execution,
                                   **kwargs
                                   )
        self.g_optimizer = self.optimizer
        self.d_optimizer = self.optimizer

    @tf.function
    def train_step(self, data):
        lr_patch, hr_patch = data
        
        # 1. update generator
        with tf.GradientTape() as tape:
            reconstruction = self.generator(lr_patch, training=True)
            disc_fake = self.discriminator(reconstruction, training=True)
            content_loss = tf.reduce_mean(
                tf.keras.losses.mean_squared_error(self.vgg(hr_patch),
                                                   self.vgg(reconstruction)
                                                   )
            )
            tv_loss = tf.reduce_mean(
                tf.image.total_variation(reconstruction)
            )
            # use -log(D(G(LR)) for faster convergence not log(1 - D(G(LR))
            adversarial_loss = -tf.losses.binary_crossentropy(tf.zeros_like(disc_fake),
                                                              disc_fake
                                                              )
            perceptual_loss = content_loss * 0.006 + adversarial_loss * 1e-03 + tv_loss * 2e-08
        grads = tape.gradient(perceptual_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        # 2. update discriminator for k times: k == 1
        with tf.GradientTape() as tape:
            reconstruction = self.generator(lr_patch, training=True)
            disc_fake = self.discriminator(reconstruction, training=True)
            disc_true = self.discriminator(hr_patch, training=True)
            discriminator_loss = - tf.losses.binary_crossentropy(tf.zeros_like(disc_true),
                                                                 disc_true
                                                                 )\
                                 + tf.losses.binary_crossentropy(tf.zeros_like(disc_fake),
                                                                 disc_fake
                                                                 )
        grads = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_variables)
        )
        psnr, ssim = compute_metrics(hr_patch, reconstruction)
        return {'content_loss': content_loss,
                'adversarial_loss': adversarial_loss,
                'discriminator_loss': discriminator_loss,
                'psnr': psnr,
                'ssim': ssim
                }

    @tf.function
    def test_step(self, data):
        lr_patch, hr_patch = data
        reconstruction = self.generator(lr_patch)
        content_loss = tf.reduce_mean(
                tf.keras.losses.mean_squared_error(self.vgg(hr_patch),
                                                   self.vgg(reconstruction)
                                                   )
            )
        adversarial_loss = -tf.losses.binary_crossentropy(tf.zeros_like(reconstruction),
                                                          reconstruction
                                                          ) * 1e-03
        psnr, ssim = compute_metrics(hr_patch, reconstruction)
        return {'content_loss': content_loss,
                'adversarial_loss': adversarial_loss,
                'psnr': psnr,
                'ssim': ssim
                }

    @tf.function
    def call(self, inputs, training=None, mask=None):
        return self.generator(inputs)
