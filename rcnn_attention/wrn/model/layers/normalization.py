import tensorflow as tf

from .utils import get_axis_internal_data_format

__all__ = ('batch_normalization',)

# -----------------------------------------------------------------------------


# This should be a partial application of tf.layers.batch_normalization, but
# only tf.contrib.layers.batch_norm supports the fused option.
def batch_normalization(
    inputs,
    axis=-1,
    momentum=0.9,
    epsilon=1e-5,
    center=True,
    scale=True,
    training=False,
    name=None,
):
    return tf.contrib.layers.batch_norm(
        inputs,
        decay=momentum,
        center=center,
        scale=scale,
        epsilon=epsilon,
        is_training=training,
        fused=True,
        data_format=get_axis_internal_data_format(axis),
        scope=name,
    )
