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


def linear(z):
    return z


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def tanh(z):
    return np.tanh(z)


def relu(z):
    return np.maximum(0., z)


def leaky_relu(z, a=0.01):
    """
    >>> leaky_relu( 1.)
    1.0
    >>> leaky_relu(-1.)
    -0.01
    """
    return np.maximum(a * z, z)


def softplus(z):
    """Smooth ReLU."""
    return np.log(1. + np.exp(z))


def softmax(z):
    """
    Examples
    --------
    >>> z = np.arange(4.)
    >>> softmax(z)
    array([[ 0.0320586 ,  0.08714432,  0.23688282,  0.64391426]])
    >>> z += 100.
    >>> softmax(z)
    array([[ 0.0320586 ,  0.08714432,  0.23688282,  0.64391426]])
    """
    z = np.atleast_2d(z)
    # avoid numerical overflow by removing max
    e = np.exp(z - np.amax(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)