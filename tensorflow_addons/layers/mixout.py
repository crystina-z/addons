"""Implementing Mixout layer."""
import numpy as np
import tensorflow as tf



@tf.keras.utils.register_keras_serializable(package="Addons")
class Mixout(tf.keras.layers.Layer):
    def __init__(self, rate, pretrain_weights):
        if not isinstance(pretrain_weights, dict):
            raise ValueError(f"Unexpect type of pretrain weights: expect dictionary mapping variable name to tf.Tensor, got {type(pretrain_weights)}")
        if not all([isinstance(value, tf.Tensor) or isinstance(value, np.array) for value in pretrain_weights.values()]):
            raise ValueError("All values in pretrain weight should be of tf.Tensor type or be convertable to tf.Tensor")
        # TODO: any existing api to do this checking?

        if (isinstance(rate, float) or isinstance(rate, int)) and rate < 0 or rate >= 1:
            raise ValueError("rate must be a scalar tensor or a float in the "
                             "range [0, 1), got %g" % rate)

        for name, variable in pretrain_weights:
            pretrain_weights[name] = tf.convert_to_tensor(variable, name=f"from_pretrain/{name}")

        self.rate = rate
        self.pretrain_weights = pretrain_weights

    def call(self, inputs):
        with tf.name_scope("mixout") as name:
            x = tf.convert_to_tensor(inputs, name="x")
            target_x = self.pretrain_weights.get(inputs.name, None)

            if not target_x:
                tf.get_logger().warn(f"pre-train weight for {inputs.name} cannot be found given weights, skip.")
                return x

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
                    ret, tgt_ret = tf.mul(x, scale), tf.mul(target_x, scale)
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
                ret = tf.real_div(x, tf.sub(one_tensor, rate))
                tgt_ret = tf.real_div(target_x, tf.sub(one_tensor, rate))

            noise_shape = tf.shape(x)
            # Sample a uniform distribution on [0.0, 1.0) and select values larger
            # than rate.
            #
            # NOTE: Random uniform can only generate 2^23 floats on [1.0, 2.0)
            # and subtract 1.0.
            random_tensor = tf.random.uniform(noise_shape, dtype=x_dtype)
            # NOTE: if (1.0 + rate) - 1 is equal to rate, then that float is selected,
            # hence a >= comparison is used.
            keep_mask = tf.cast(random_tensor >= rate, x_dtype)
            replace_mask = tf.cast(random_tensor < rate, x_dtype)
            ret = tf.add(tf.mul(ret, keep_mask), tf.mul(tgt_ret, replace_mask))
            ret = ret * keep_mask + tgt_ret * replace_mask - rate * tgt_ret

            if not is_executing_eagerly:
                ret.set_shape(x.get_shape())
            return ret
