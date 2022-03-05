import tensorflow as tf
import tensorflow_addons as tfa


class BiocubicUpsampling2D(tf.keras.layers.Layer):
    def __init__(self, rate):
        super(BiocubicUpsampling2D, self).__init__()
        self.rate = rate

    def call(self, inputs, *args, **kwargs):
        b, h, w, c = inputs.get_shape().as_list()
        return tf.image.resize(inputs,
                               (h * self.rate, w * self.rate),
                               method=tf.image.ResizeMethod.BICUBIC
                               )
