import functools

import tensorflow as tf

#import dl_papers.common.layers as dl_layers
import rcnn_attention.wrn.model.layers as dl_layers
# -----------------------------------------------------------------------------


def wide_resnet_model(
    inputs,
    depth,
    width_factor,
    dropout_rate=0,
    scalar_gate=False,
    data_format='channels_last',
    training=False,
):
    assert (depth - 4) % 6 == 0, "impossible network depth"

    conv2d = functools.partial(
        dl_layers.resnet.conv2d,
        data_format=data_format,
    )

    residual_group = functools.partial(
        dl_layers.resnet.residual_group,
        num_layers=(depth - 4) / 3,
        dropout_rate=dropout_rate,
        scalar_gate=scalar_gate,
        data_format=data_format,
        training=training,
    )

    batch_normalization = functools.partial(
        dl_layers.batch_normalization,
        axis=dl_layers.get_channel_axis(data_format),
        training=training,
    )

    global_avg_pooling2d = functools.partial(
        tf.reduce_mean,
        axis=dl_layers.get_spatial_axes(data_format),
    )

    net = inputs
    activations = {}
    net = conv2d(net, 16, 3, name='pre_conv')
    activations['pre_conv'] = net
    net = residual_group(
        net,
        filters=16 * width_factor,
        strides=1,
        name='group_1',
    )
    activations['group_1'] = net
    net = residual_group(
        net,
        filters=32 * width_factor,
        strides=2,
        name='group_2',
    )
    activations['group_2'] = net
    net = residual_group(
        net,
        filters=64 * width_factor,
        strides=2,
        name='group_3',
    )
    activations['group_3'] = net

    net = batch_normalization(net, name='post_bn')
    net = tf.nn.relu(net, name='post_relu')
    activations['post_relu'] = net
    net = global_avg_pooling2d(net, name='post_pool')

    return net, activations


# -----------------------------------------------------------------------------

#wide_resnet_cifar10 = functools.partial(
#    wide_resnet_cifar,
#    depth=28,
#    width_factor=4,
#)

#wide_gated_resnet_cifar10 = functools.partial(
#    wide_resnet_cifar10,
#    scalar_gate=True,
#)
