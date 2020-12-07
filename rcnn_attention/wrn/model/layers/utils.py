import functools

import tensorflow as tf

__all__ = (
    'with_name_scope',
    'with_variable_scope',
    'get_channel_axis',
    'get_spatial_axes',
    'get_axis_internal_data_format',
)

# -----------------------------------------------------------------------------


def _get_inputs(inputs, *args, **kwargs):
    return (inputs,)


def with_name_scope(func=None, get_values=_get_inputs):
    if func is None:
        return functools.partial(with_name_scope, get_values=get_values)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        name = kwargs.pop('name', None)

        with tf.name_scope(
            name, func.__name__, get_values(*args, **kwargs),
        ):
            return func(*args, **kwargs)

    return wrapped


def with_variable_scope(func=None, get_values=_get_inputs):
    if func is None:
        return functools.partial(with_variable_scope, get_values=get_values)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        name = kwargs.pop('name', None)
        reuse = kwargs.pop('reuse', None)

        with tf.variable_scope(
            name, func.__name__, get_values(*args, **kwargs), reuse=reuse,
        ):
            return func(*args, **kwargs)

    return wrapped


# -----------------------------------------------------------------------------


def get_channel_axis(data_format):
    return 1 if data_format == 'channels_first' else -1


def get_spatial_axes(data_format):
    return (2, 3) if data_format == 'channels_first' else (1, 2)


def get_axis_internal_data_format(axis):
    assert axis in (-1, 1, 3)
    return 'NCHW' if axis == 1 else 'NHWC'
