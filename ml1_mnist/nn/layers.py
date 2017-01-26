import numpy as np

from activations import get_activation
from initializations import get_initialization
import env
from utils import RNG


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
    def __init__(self, output_dim, bias=1.0, init='glorot_uniform', L1=0.0, L2=0.0, max_norm=-1, **params):
        self.output_dim = output_dim
        self.bias = bias
        self.init = get_initialization(init)
        self.L1 = L1
        self.L2 = L2
        self.max_norm = max_norm
        self.W = np.array([]) # weights will be updated by optimizer
        self.b = np.array([])
        self.dW = np.array([]) # dW, db will be used by optimizer
        self.db = np.array([])
        self._last_input = None
        super(FullyConnected, self).__init__(**params)

    def _max_norm_update(self):
        L = np.linalg.norm(self.W, np.inf)
        if L > self.max_norm > 0:
            self.W *= self.max_norm / L

    def setup_weights(self, x_shape):
        self.W = self.init(shape=(x_shape[1], self.output_dim), random_seed=self.random_seed)
        self.b = np.full(self.W.shape[1], self.bias)

    def forward_pass(self, x):
        self._last_input = x
        return np.dot(x, self.W) + self.b

    def backward_pass(self, residual):
        self.dW = np.dot(self._last_input.T, residual) + self.L2 * self.W + self.L1 * np.sign(self.W)
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


class Dropout(BaseLayer):
    """
    Randomly set a fraction of `p` inputs to 0
    at each training update.
    """
    def __init__(self, p=0.2, **params):
        self.p = p
        self.is_training = True
        self._mask = None
        super(Dropout, self).__init__(**params)

    def forward_pass(self, X):
        assert self.p > 0
        if self.is_training:
            self._mask = RNG(self.random_seed).uniform(size=X.shape) > self.p
            Z = self._mask * X
        else: # to keep output of the same scale (on average)
            Z = (1.0 - self.p) * X
        return Z

    def backward_pass(self, residual):
        return self._mask * residual


# aliases
Dense = FullyConnected


if __name__ == '__main__':
    pass