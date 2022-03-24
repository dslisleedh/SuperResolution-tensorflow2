from utils import *
import tensorflow as tf


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
            BicubicScale2D(self.expansion_rate),
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
                                   strides=1,
                                   activation='linear'
                                   )
        ])

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            pred = self.forward(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, pred))
        grads = tape.gradient(loss, self.forward.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.forward.trainable_variables)
        )
        psnr, ssim = compute_metrics(pred, y)
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}

    @tf.function
    def test_step(self, data):
        x, y = data
        pred = self.forward(x)
        loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, pred))
        psnr, ssim = compute_metrics(pred, y)
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}

    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)
