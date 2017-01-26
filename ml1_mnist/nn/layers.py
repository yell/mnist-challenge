import numpy as np

from activations import get_activation
from initializations import get_initialization


class BaseLayer(object):
    def __init__(self, random_seed=None):
        self.random_seed = random_seed

    def setup_weights(self, x_shape):
        pass

    def forward_pass(self, x):
        raise NotImplementedError()

    def backward_pass(self, residual):
        raise NotImplementedError()

    def shape(self, prev_shape):
        return prev_shape

    @property
    def n_params(self):
        return 0


class FullyConnected(BaseLayer):
    """
    Examples
    --------
    >>> fc = FullyConnected(10)
    >>> fc.setup_weights(x_shape=(128, 32)) # (batch_size, n_features)
    >>> fc.W.shape
    (32, 10)
    >>> fc.b.shape
    (10,)
    >>> fc.n_params # size of W + size of b
    330
    """
    def __init__(self, output_dim, bias=1.0, init='glorot_uniform', L2=0.0, **params):
        self.output_dim = output_dim
        self.bias = bias
        self.init = get_initialization(init)
        self.L2 = L2
        self.W = np.array([]) # weights will be updated by optimizer
        self.b = np.array([])
        self.dW = np.array([]) # dW, db will be used by optimizer
        self.db = np.array([])
        self._last_input = None
        super(FullyConnected, self).__init__(**params)

    def setup_weights(self, x_shape):
        self.W = self.init(shape=(x_shape[1], self.output_dim), random_seed=self.random_seed)
        self.b = np.full(self.W.shape[1], self.bias)

    def forward_pass(self, x):
        self._last_input = x
        return np.dot(x, self.W) + self.b

    def backward_pass(self, residual):
        self.dW = np.dot(self._last_input.T, residual) + self.L2 * self.W
        self.db = np.sum(residual, axis=0)
        return np.dot(residual, self.W.T)

    def shape(self, prev_shape):
        return prev_shape[1], self.output_dim

    @property
    def n_params(self):
        return np.size(self.W) + np.size(self.b)


class Activation(BaseLayer):
    def __init__(self, activation, **params):
        self.activation = get_activation(activation)
        self._last_input = None
        super(Activation, self).__init__(**params)

    def forward_pass(self, x):
        self._last_input = x
        return self.activation(x)

    def backward_pass(self, residual):
        return self.activation(self._last_input, derivative=True) * residual


# aliases
Dense = FullyConnected


if __name__ == '__main__':
    pass