import numpy as np
import scipy.spatial.distance as dist


def get_kernel(kernel_name, **kernel_params):
    """
    Examples
    --------
    >>> get_kernel('rbf', gamma=2.)
    1.0 ** 2 * RBF(gamma=2.0)
    """
    for k, v in globals().items():
        if k.lower() == kernel_name.lower():
            return v(**kernel_params)
    raise ValueError("invalid kernel name '{0}'".format(kernel_name))


class BaseKernel(object):
    def __call__(self, x, y):
        """Compute the kernel function on `x`, `y`.

        Parameters
        ----------
        x, y : floats or (N,) | (M,N) np.ndarray
            Input vectors.

        Returns
        -------
        result : float or np.ndarray
            Computed value of kernel.
        """
        _x, _y = self._check_x_y(x, y)
        result = self._call(_x, _y)
        if len(result.shape) == 1 and result.shape[0] == 1 or \
                                len(result.shape) == 2 and result.shape == (1, 1):
            return np.asscalar(result)
        return result

    def __repr__(self):
        return self.__class__.__name__

    def _check_x_y(self, x, y):
        return np.atleast_2d(x), np.atleast_2d(y)

    def _call(self, x, y):
        raise NotImplementedError()


class Linear(BaseKernel):
    """Linear kernel:

    k(x, y) = (x, y)

    Examples
    --------
    >>> linear = Linear()
    >>> linear(0., 1.)
    0.0
    >>> linear([0., 1.], [0.5, 2.])
    2.0
    >>> linear([[0.], [1.]], [[0.5], [2.]])
    array([[ 0. ,  0. ],
           [ 0.5,  2. ]])
    >>> linear([[0., 1.], [0.5, 2.]], [0.5, 1.])
    array([[ 1.  ],
           [ 2.25]])
    """
    def __init__(self, **kwargs):
        pass

    def _call(self, x, y):
        return np.dot(x, y.T)


class Poly(BaseKernel):
    """Polynomial kernel:

    k(x, y) = (`gamma` * (x, y) + `offset`) ** `degree`,
    `gamma` > 0

    Examples
    --------
    >>> poly = Poly(offset=1.)
    >>> poly(1., 2.)
    9.0
    >>> poly([0., 1.], [0.5, 2.])
    9.0
    >>> poly([[0.], [1.]], [[0.5], [2.]])
    array([[ 1.  ,  1.  ],
           [ 2.25,  9.  ]])
    >>> poly([[0., 1.], [0.5, 2.]], [0.5, 1.])
    array([[  4.    ],
           [ 10.5625]])
    """
    def __init__(self, degree=2.0, offset=0.0, gamma=1.0, **kwargs):
        self.degree = degree
        self.offset = offset
        self.gamma = gamma

    def __repr__(self):
        s = super(Poly, self).__repr__()
        s += "(degree={0}, offset={1}, gamma={2})" \
            .format(self.degree, self.offset, self.gamma)
        return s

    def _call(self, x, y):
        return (self.gamma * np.dot(x, y.T) + self.offset) ** self.degree


class RBF(BaseKernel):
    """RBF kernel:

    k(x, y) = `sigma` ** 2 * exp(-`gamma` * ||x - y|| ** 2),
    `gamma`, `sigma` > 0

    Examples
    --------
    >>> rbf = 4. * RBF()
    >>> rbf
    2.0 ** 2 * RBF(gamma=1.0)
    >>> rbf(0., 1.)
    1.4715177646857693
    >>> rbf([0., 1.], [0.5, 2.])
    1.1460191874407601
    >>> rbf([[0.], [1.]], [[0.5], [2.]])
    array([[ 3.11520313,  0.07326256],
           [ 3.11520313,  1.47151776]])
    >>> rbf([[0., 1.], [0.5, 2.]], [0.5, 1.])
    array([[ 3.11520313],
           [ 1.47151776]])
    """
    def __init__(self, gamma=1.0, sigma=1.0, **kwargs):
        self.gamma = gamma
        self.sigma = sigma

    def __repr__(self):
        s = "{0} ** 2 * ".format(self.sigma)
        s += super(RBF, self).__repr__()
        s += "(gamma={0})".format(self.gamma)
        return s

    def __rmul__(self, sigma2):
        return RBF(gamma=self.gamma, sigma=self.sigma * np.sqrt(sigma2))

    def _call(self, x, y):
        return self.sigma ** 2 * np.exp(-self.gamma * dist.cdist(x, y) ** 2)


class Sigmoid(BaseKernel):
    """Sigmoid kernel:

    k(x, y) = tanh(`gamma` * (x,y) + `offset`),
    `gamma` > 0

    Examples
    --------
    >>> sigm = Sigmoid()
    >>> sigm(0., 1.)
    0.7615941559557649
    >>> sigm([0., 1.], [0.5, 2.])
    0.8068839875063545
    >>> sigm([[0.], [1.]], [[0.5], [2.]])
    array([[ 0.46211716,  0.96402758],
           [ 0.46211716,  0.76159416]])
    >>> sigm([[0., 1.], [0.5, 2.]], [0.5, 1.])
    array([[ 0.46211716],
           [ 0.76159416]])
    """
    def __init__(self, gamma=1.0, offset=0.0, **kwargs):
        self.gamma = gamma
        self.offset = offset

    def __repr__(self):
        s = super(RBF, self).__repr__()
        s += "(gamma={0}, offset={1})".format(self.gamma)

    def _call(self, x, y):
        return np.tanh(self.gamma * dist.cdist(x, y) + self.offset)


if __name__ == '__main__':
    # run corresponding tests
    from utils.testing import run_tests
    run_tests(__file__)