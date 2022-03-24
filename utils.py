from einops import rearrange
import tensorflow as tf


@tf.function
def compute_metrics(label, pred):
    psnr = tf.reduce_mean(tf.image.psnr(pred, label, 1.))
    ssim = tf.reduce_mean(tf.image.ssim(pred, label, 1.))
    return psnr, ssim


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
            shape = tf.shape(inputs)
            h = shape[1]
            w = shape[2]
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


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, strides):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.strides = strides

    def call(self, images):
        b, h, w, c = images.get_shape().as_list()
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.strides, self.strides, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = rearrange(patches, 'b hp wp (p1 p2 c) -> (b hp wp) p1 p2 c',
                            b=b, p1=self.patch_size, p2=self.patch_size, c=c
                            )
        return patches


class DegradationLayer(tf.keras.layers.Layer):
    def __init__(self, noise_max=25 / 255, blur_sigma=10, scale_rate=4):
        super(DegradationLayer, self).__init__()
        self.noise = noise_max
        self.blur_sigma = blur_sigma
        self.scale_rate = scale_rate

        self.blur = GaussianBlur(7, self.blur_sigma)

    def call(self, inputs, *args, **kwargs):
        b, h, w, c = tf.shape(inputs)
        lr = self.blur(inputs)
        lr = tf.image.resize(lr,
                             (h // self.scale_rate, w // self.scale_rate),
                             method=tf.image.ResizeMethod.BICUBIC
                             )
        noise = tf.random.normal(shape=(b, h // self.scale_rate, w // self.scale_rate, c),
                                 mean=0.,
                                 stddev=self.noise
                                 )
        lr = tf.clip_by_value(lr + noise,
                              clip_value_min=0.,
                              clip_value_max=1.
                              )
        return lr


class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self,
                 scale_rate
                 ):
        super(PixelShuffle, self).__init__()
        self.scale_rate = scale_rate

    def call(self, inputs, *args, **kwargs):
        return tf.nn.depth_to_space(inputs,
                                    block_size=self.scale_rate
                                    )

