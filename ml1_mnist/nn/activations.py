import numpy as np


def get_activation(activation_name):
    """
    Examples
    --------
    >>> get_activation('softplus')(1.) # doctest: +ELLIPSIS
    1.313...
    """
    for k, v in globals().items():
        if k.lower() == activation_name.lower():
            return v
    raise ValueError("invalid activation function name '{0}'".format(activation_name))


def linear(z, derivative=False):
    if derivative:
        return np.ones_like(z)
    return z


def sigmoid(z, derivative=False):
    y = 1. / (1. + np.exp(-z))
    if derivative:
        return y * (1. - y) # TODO: or z * (1. - z)?
    return y


def tanh(z, derivative=False):
    y = np.tanh(z)
    if derivative:
        return 1. - y ** 2 # TODO: or 1. - z ** 2?
    return y


def relu(z, derivative=False):
    """
    Examples
    --------
    >>> z = [[-1, 2,   -3, 4, 5],
    ...      [ 1, 2, -0.1, 3, 4]]
    >>> relu(z)
    array([[ 0.,  2.,  0.,  4.,  5.],
           [ 1.,  2.,  0.,  3.,  4.]])
    >>> relu(z, derivative=True)
    array([[ 0.,  1.,  0.,  1.,  1.],
           [ 1.,  1.,  0.,  1.,  1.]])
    """
    if derivative:
        z = np.asarray(z)
        d = np.zeros_like(z)
        d[z > 0] = 1.
        return d
    return np.maximum(0., z)


def leaky_relu(z, a=0.01, derivative=False):
    """
    >>> leaky_relu( 1.)
    1.0
    >>> leaky_relu(-1.)
    -0.01
    """
    if derivative:
        z = np.asarray(z)
        d = np.ones_like(z)
        d[z < 0] = a
        return d
    return np.maximum(a * z, z)


def softplus(z, derivative=False):
    # TODO: prevent overflow
    """Smooth ReLU."""
    if derivative:
        return sigmoid(z)
    return np.log(1. + np.exp(z))


def softmax(z, derivative=False):
    """
    Examples
    --------
    >>> z = np.log([1, 2, 5])
    >>> softmax(z)
    array([[ 0.125,  0.25 ,  0.625]])
    >>> z += 100.
    >>> softmax(z)
    array([[ 0.125,  0.25 ,  0.625]])
    >>> softmax(z, derivative=True)
    array([[ 0.109375,  0.1875  ,  0.234375]])
    """
    z = np.atleast_2d(z)
    # avoid numerical overflow by removing max
    e = np.exp(z - np.amax(z, axis=1, keepdims=True))
    y = e / np.sum(e, axis=1, keepdims=True)
    if derivative: # TODO: or z * (1. - z)?
        return y * (1. - y) # element-wisely
    return y