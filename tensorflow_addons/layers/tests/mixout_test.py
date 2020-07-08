import pytest
import numpy as np

import tensorflow as tf
from tensorflow_addons.layers.mixout import Mixout
from tensorflow_addons.utils import test_utils


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_mixout(dtype):
    x = tf.ones([3,5], name="x")
    target_x = tf.ones_like(x) * 2
    pretrained_weight = {"x": target_x}
    expected_output = np.asarray(
        [[3., 1., 3., 1., 1.], [3., 3., 3., 1., 1.], [1., 3., 3., 3., 3.]], dtype=np.float32)

    output = test_utils.layer_test(
        Mixout,
        kwargs={"rate": 0.5, "pretraiend_weight": pretrained_weight},
        input_data=x,
        expected_output=expected_output
    )
