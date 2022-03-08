import tensorflow as tf


@tf.function
def compute_psnr(pred, label):
    return tf.reduce_mean(tf.image.psnr(pred, label, 1.))


@tf.function
def compute_ssim(pred, label):
    return tf.reduce_mean(tf.image.ssim(pred, label, 1.))


class BicubicScale2D(tf.keras.layers.Layer):
    def __init__(self,
                 rate,
                 input_size=None
                 ):
        super(BicubicScale2D, self).__init__()
        self.rate = rate
        self.input_size = input_size

    def call(self, inputs, *args, **kwargs):
        if self.input_size is None:
            _, h, w, _ = inputs.get_shape().as_list()
        else:
            h, w = self.input_size
        return tf.image.resize(inputs,
                               (int(h * self.rate), int(w * self.rate)),
                               method=tf.image.ResizeMethod.BICUBIC
                               )


class GaussianBlur:
    def __init__(self, kernel_size, sigma, n_channels=3):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.n_channels = n_channels

        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * self.sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, self.n_channels])
        self.gaussian_kernel = kernel[..., tf.newaxis]

    @tf.function
    def __call__(self, img):
        if len(img.get_shape()) == 3:
            return tf.nn.depthwise_conv2d(tf.expand_dims(img,
                                                         axis=0
                                                         ),
                                          self.gaussian_kernel,
                                          [1, 1, 1, 1],
                                          padding='SAME'
                                          )
        else:
            return tf.nn.depthwise_conv2d(img,
                                          self.gaussian_kernel,
                                          [1, 1, 1, 1],
                                          padding='SAME'
                                          )
