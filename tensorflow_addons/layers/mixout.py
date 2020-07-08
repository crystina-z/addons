"""Implementing Mixout layer."""
import numpy as np
import tensorflow as tf



@tf.keras.utils.register_keras_serializable(package="Addons")
class Mixout(tf.keras.layers.Layer):
    """Computes mixout.

    "Mixout: Effective regularization to finetune large-scale pretrained language models."
    Lee, Cheolhyoung, Kyunghyun Cho, and Wanmo Kang.
    arXiv:1909.11299 (2019)

    For each element of `inputs`, with probability `rate`, outputs the corresponding weight in
    `pretraine_weights[inputs.name]`, then the replaced tensor scales up the input by `1 / (1-rate)`.
    The scaling is such that the expected sum is unchanged.

    Argument:
      seed: A Python integer. Used to create random seeds. See `tf.random.set_seed` for behavior.
      pretrain_weights: A Python dictionary. Store the mapping from Variables' names to their
        corresponding pre-trained weight. Each pre-trained weight must be of same shape with
        the current Variable.
      rate: A scalar or scalar `Tensor` with the same type as `x`. The probability that each
        element of `x` is discarded.

    Input shape:
        nD trainable Variable `(D1, D2, .., Dn)`

    Output shape:
        nD trainable Variable `(D1, D2, .., Dn)`
    """
    def __init__(self, rate, pretrain_weights, seed=None, **kwargs):
        super(Mixout, self).__init__()
        if not isinstance(pretrain_weights, dict):
            raise ValueError(f"Unexpect type of pretrain weights: expect dictionary mapping variable name to tf.Tensor, got {type(pretrain_weights)}")
        if not all([isinstance(value, tf.Tensor) or isinstance(value, np.array) for value in pretrain_weights.values()]):
            raise ValueError("All values in pretrain weight should be of tf.Tensor type or be convertable to tf.Tensor")
        # TODO: any existing api to do this checking?

        if (isinstance(rate, float) or isinstance(rate, int)) and rate < 0 or rate >= 1:
            raise ValueError("rate must be a scalar tensor or a float in the "
                             "range [0, 1), got %g" % rate)

        for name, variable in pretrain_weights.items():
            pretrain_weights[name] = tf.convert_to_tensor(variable, name=f"from_pretrain/{name}")

        self.rate = rate
        self.seed = seed
        self.pretrain_weights = pretrain_weights

    def call(self, inputs):
        with tf.name_scope("mixout") as name:
            x = tf.convert_to_tensor(inputs, name="x")

            if inputs.name not in self.pretrain_weights:
                tf.get_logger().warn(f"pre-train weight for {inputs.name} cannot be found given weights, skip.")
                return x

            target_x = self.pretrain_weights[inputs.name]
            x_dtype = x.dtype
            if not x_dtype.is_floating:
                raise ValueError("x has to be a floating point tensor since it's going "
                                 "to be scaled. Got a %s tensor instead." % x_dtype)
            if x_dtype != target_x.dtype:
                raise ValueError("x and target x have to have same dtype. Got %s as x's "
                                 "dtype and %s as target x's dtype" % (x_dtype, target_x.dtype))

            is_executing_eagerly = tf.executing_eagerly()
            if not tf.is_tensor(self.rate):
                if isinstance(self.rate, float) or isinstance(self.rate, int):
                    keep_prob = 1 - self.rate
                    scale = 1 / keep_prob
                    scale = tf.convert_to_tensor(scale, dtype=x_dtype)
                    ret, tgt_ret = tf.multiply(x, scale), tf.multiply(target_x, scale)
                else:
                    raise ValueError("rate is neither scalar nor scalar tensor %r" % self.rate)
            else:
                self.rate.get_shape().assert_has_rank(0)
                rate_dtype = self.rate.dtype
                if rate_dtype != x_dtype:
                    if not rate_dtype.is_compatible_with(x_dtype):
                        raise ValueError(
                            "Tensor dtype %s is incomptaible with Tensor dtype %s: %r" %
                            (x_dtype.name, rate_dtype.name, self.rate))
                    rate = tf.cast(self.rate, x_dtype, name="rate")
                one_tensor = tf.constant(1, dtype=x_dtype)
                ret = tf.real_div(x, tf.subtract(one_tensor, rate))
                tgt_ret = tf.real_div(target_x, tf.subtract(one_tensor, rate))

            noise_shape = tf.shape(x)
            # Sample a uniform distribution on [0.0, 1.0) and select values larger
            # than rate.
            #
            # NOTE: Random uniform can only generate 2^23 floats on [1.0, 2.0)
            # and subtract 1.0.
            random_tensor = tf.random.uniform(noise_shape, seed=self.seed, dtype=x_dtype)
            # NOTE: if (1.0 + rate) - 1 is equal to rate, then that float is selected,
            # hence a >= comparison is used.
            keep_mask = tf.cast(random_tensor >= self.rate, x_dtype)
            replace_mask = tf.cast(random_tensor < self.rate, x_dtype)
            ret = tf.add(tf.multiply(ret, keep_mask), tf.multiply(tgt_ret, replace_mask))
            ret = tf.subtract(ret, tf.multiply(self.rate, tgt_ret))

            if not is_executing_eagerly:
                ret.set_shape(x.get_shape())
            return ret
