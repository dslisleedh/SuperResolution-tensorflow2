import tensorflow as tf
from utils import *


'''
https://arxiv.org/abs/1511.04587
use MSE to train
VGG-like architecture
gradient clipping/high learning rate is applied
use residual learning(**NOT ResNet-like architecture**)
'''
class VDSR(tf.keras.models.Model):
    def __init__(self,
                 theta,
                 n_filters=64,
                 n_layers=20,
                 scale_rate=4,
                 out_channels=3
                 ):
        super(VDSR, self).__init__()
        self.theta = theta
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.scale_rate = scale_rate
        self.out_channels = out_channels

        self.upscaler = BicubicScale2D(self.scale_rate)
        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.out_channels if i == (self.n_layers - 1) else self.n_filters,
                                   kernel_size=3,
                                   strides=1,
                                   padding='SAME',
                                   activation='linear' if i == (self.n_layers - 1) else 'relu',
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=tf.keras.regularizers.l2(l2=.0001)
                                   ) for i in range(self.n_layers)
        ])

    @tf.function
    def train_step(self, data):
        x, y = data
        x = self.upscaler(x)
        with tf.GradientTape() as tape:
            l2regularization = tf.reduce_sum(self.losses)
            pred = self.forward(x, training=True) + x
            loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, pred))
            total_loss = loss + l2regularization
        grads = tape.gradient(total_loss, self.forward.trainable_variables)
        self.optimizer.apply_gradients(
            zip([tf.clip_by_value(g,
                                  -self.theta/self.optimizer.learning_rate,
                                  self.theta/self.optimizer.learning_rate
                                  ) for g in grads],
                self.forward.trainable_variables
                )
        )
        psnr, ssim = compute_metrics(pred, y)
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}

    @tf.function
    def test_step(self, data):
        x, y = data
        x = self.upscaler(x)
        pred = self.forward(x) + x
        loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, pred))
        psnr, ssim = compute_metrics(pred, y)
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}

    @tf.function
    def call(self, inputs, training=None, mask=None):
        inputs = self.upscaler(inputs)
        return self.forward(inputs) + inputs

