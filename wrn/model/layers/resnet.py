import functools

import tensorflow as tf

from .normalization import batch_normalization
from .utils import get_channel_axis, with_variable_scope

# -----------------------------------------------------------------------------

conv2d = functools.partial(
    tf.layers.conv2d,
    padding='same',
    use_bias=False,
)

# -----------------------------------------------------------------------------


@with_variable_scope
def bn_relu_conv(
    net,
    filters,
    kernel_size,
    bn_momentum=0.9,
    dropout_rate=0,
    data_format='channels_last',
    training=False,
    **kwargs
):
    net = batch_normalization(
        net,
        axis=get_channel_axis(data_format),
        momentum=bn_momentum,
        training=training,
    )
    net = tf.nn.relu(net)

    if dropout_rate != 0:
        net = tf.layers.dropout(net, rate=dropout_rate, training=training)

    net = conv2d(
        net,
        filters,
        kernel_size,
        data_format=data_format,
        **kwargs
    )

    return net


@with_variable_scope
def scalar_gating(
    net,
    activation=tf.nn.relu,
    k_initializer=tf.ones_initializer(),
    k_regularizer=None,
    k_regularizable=False,
):
    # Represent this with shape (1,) instead of as a scalar to get proper
    # parameter count from tfprof.
    k = tf.get_variable(
        'k',
        (1,),
        initializer=k_initializer,
        regularizer=k_regularizer,
        trainable=True,
    )

    # Per the paper, we may specifically not want to regularize k.
    k.regularizable = k_regularizable

    return activation(k) * net


@with_variable_scope
def residual_group(
    net,
    num_layers,
    filters,
    strides=1,
    bn_momentum=0.9,
    dropout_rate=0,
    no_preact=False,
    scalar_gate=False,
    data_format='channels_last',
    training=False,
):
    assert num_layers % 2 == 0, "impossible number of layers"

    channel_axis = get_channel_axis(data_format)

    batch_normalization_bound = functools.partial(
        batch_normalization,
        axis=channel_axis,
        momentum=bn_momentum,
        training=training,
    )

    strided_conv2d = functools.partial(
        conv2d,
        filters=filters,
        strides=strides,
        data_format=data_format,
    )

    bn_relu_conv_bound = functools.partial(
        bn_relu_conv,
        filters=filters,
        kernel_size=3,
        bn_momentum=bn_momentum,
        data_format=data_format,
        training=training,
    )

    for i in range(num_layers // 2):
        use_projection_shortcut = (
            i == 0 and (
                strides != 1 or
                filters != net.get_shape()[channel_axis].value
            )
        )

        if no_preact:
            layer = strided_conv2d(net, kernel_size=3)
        elif use_projection_shortcut:
            net = batch_normalization_bound(net)
            net = tf.nn.relu(net)

            layer = strided_conv2d(net, kernel_size=3)
        else:
            layer = bn_relu_conv_bound(net)

        layer = bn_relu_conv_bound(layer, dropout_rate=dropout_rate)

        if scalar_gate:
            layer = scalar_gating(layer)

        if use_projection_shortcut:
            net = strided_conv2d(net, kernel_size=1)

        net += layer

    return net
