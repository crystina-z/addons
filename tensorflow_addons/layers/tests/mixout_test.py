import pytest
import numpy as np

import tensorflow as tf
from tensorflow_addons.layers.mixout import Mixout
from tensorflow_addons.utils import test_utils


@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.usefixtures("set_seeds")
@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_mixout(dtype):
    x = tf.Variable(tf.ones([3, 5]), name="x")  # x should be trainable weight
    target_x = tf.ones_like(x) * 2
    pretrained_weights = {x.name: target_x}

    rate_expected = [
        (0, x.numpy()),
        (0.5, np.asarray([[2., 0., 0., 2., 0.], [2., 2., 0., 2., 2.], [2., 0., 2., 0., 2.]], dtype=np.float32)),
        (0.8, np.asarray([[2., -3., -3., -3., 2.], [2., 2., -3., 2., -3.], [2., 2., 2., 2., 2.]], dtype=np.float32)),
    ]

    for rate, expected_output in rate_expected:
        mixout = Mixout(rate=rate, pretrain_weights=pretrained_weights, seed=0)
        assert (expected_output == mixout.call(x).numpy()).all()

    # output = test_utils.layer_test(
    #     Mixout,
    #     kwargs={"rate": 0.5, "pretrain_weights": pretrained_weights},
    #     input_data=x,
    #     expected_output=expected_output
    # )
