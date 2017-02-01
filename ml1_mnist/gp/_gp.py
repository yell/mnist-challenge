import numpy as np
from scipy.linalg import cholesky, solve_triangular

import env
from base import BaseEstimator
from kernels import get_kernel
from nn.activations import softmax


def log_sum_exp(x):
    """Compute log(sum(exp(x))) in a numerically stable way.

    Examples
    --------
    >>> x = [0, 1, 0]
    >>> log_sum_exp(x) #doctest: +ELLIPSIS
    1.551...
    >>> x = [1000, 1001, 1000]
    >>> log_sum_exp(x) #doctest: +ELLIPSIS
    1001.551...
    >>> x = [-1000, -999, -1000]
    >>> log_sum_exp(x) #doctest: +ELLIPSIS
    -998.448...
    """
    x = np.asarray(x)
    a = max(x)
    return a + np.log(sum(np.exp(x - a)))


class GPClassifier(BaseEstimator):
    """
    Gaussian processes classificator (GPC).

    Parameters
    ----------
    kernel : {'rbf'}, optional
        Specifies the kernel type to be used in the algorithm.
        Currently only rbf is supported.
    kernel_params : dict, optional
        Initial params of the `kernel`.
    max_iter : positive int, optional
        Maximum number of Newton iterations.
    tol : positive float, optional
        Tolerance for approx. LML for Newton iterations.
    random_seed : None or int, optional
        Pseudo-random number generator seed used for random sampling.

    Attributes
    ----------
    K_
        Covariance function.
    f_ : (n_samples, n_classes)
        Posterior approximation mode.
    lml_ : float
        Approx. log marginal likelihood \log{q(y|X, theta)},
        where theta are kernel parameters

    Examples
    --------
    >>> from utils import one_hot, one_hot_decision_function
    >>> from metrics import accuracy_score, log_loss
    >>> from nn.activations import softmax
    >>> X = [[0., 0.],
    ...      [0., 1.],
    ...      [1., 0.],
    ...      [1., 1.]]
    >>> y = one_hot([0, 1, 1, 0])
    >>> gp = GPClassifier(kernel_params=dict(sigma=1., gamma=1.))
    >>> gp.fit(X, y).K_
    array([[ 1.        ,  0.36787944,  0.36787944,  0.13533528],
           [ 0.36787944,  1.        ,  0.13533528,  0.36787944],
           [ 0.36787944,  0.13533528,  1.        ,  0.36787944],
           [ 0.13533528,  0.36787944,  0.36787944,  1.        ]])
    >>> pi = softmax(gp.f_); pi
    array([[ 0.58587507,  0.41412493],
           [ 0.41451392,  0.58548608],
           [ 0.41448831,  0.58551169],
           [ 0.58519245,  0.41480755]])
    >>> y_pred = one_hot_decision_function(pi); y_pred
    array([[ 1.,  0.],
           [ 0.,  1.],
           [ 0.,  1.],
           [ 1.,  0.]])
    >>> accuracy_score(y, y_pred)
    1.0
    >>> log_loss(y, pi) #doctest: +ELLIPSIS
    0.535...
    >>> gp.lml_ #doctest: +ELLIPSIS
    -3.995...

    >>> from utils.dataset import load_mnist
    >>> from model_selection import TrainTestSplitter as TTS
    >>> X, y = load_mnist('train', '../../data/')
    >>> train, _ = TTS(random_seed=1337, shuffle=True).split(y, train_ratio=0.0015, stratify=True)
    >>> X = X[train]; X.shape
    (84, 784)
    >>> y = one_hot(y[train])
    >>> X /= 255.
    >>> pi = softmax(gp.fit(X, y).f_);
    >>> accuracy_score(y, one_hot_decision_function(pi))
    1.0
    >>> log_loss(y, pi) #doctest: +ELLIPSIS
    1.645...
    >>> gp.lml_ #doctest: +ELLIPSIS
    -200.76...
    """

    def __init__(self, kernel='rbf', kernel_params={},
                 max_iter=100, tol=1e-4,
                 random_seed=None):
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.max_iter = max_iter
        self.tol = tol
        self.random_seed = random_seed

        self.K_ = None
        self.f_ = None
        self.lml_ = -np.inf

        super(GPClassifier, self).__init__(_y_required=True)

    def _fit(self, X, y):
        """
        Algorithm (3.3) from GPML
        with shared covariance matrix among all latent functions.
        """
        self._kernel = get_kernel(self.kernel, **self.kernel_params)

        # shortcuts
        C = self._n_outputs
        n = self._n_samples

        # construct covariance matrix
        self.K_ = self._kernel(X, X)

        # init latent function values
        self.f_ = np.zeros_like(y)

        lmls = []
        iter = 0
        while True:
            iter += 1
            if iter > self.max_iter:
                raise RuntimeError('convergence is not reached')
            self.pi_ = softmax(self.f_)
            z = []
            E = []
            for c_ in xrange(C):
                sqrt_d_c = np.sqrt(self.pi_[:, c_])
                sqrt_D_c = np.diag(sqrt_d_c)
                # same as I + sqrt_D.dot(K).dot(sqrt_D):
                _T = np.eye(self._n_samples) + (sqrt_d_c * self.K_.T).T * sqrt_d_c
                L = cholesky(_T, lower=True, overwrite_a=True)
                _T2 = solve_triangular(L, sqrt_D_c)
                E_c = sqrt_D_c.dot(solve_triangular(L, _T2, trans='T'))
                E.append(E_c)
                z_c = sum(np.log(L.diagonal()))
                z.append(z_c)
            M = cholesky(sum(E), lower=True, overwrite_a=True)
            D = np.diag(self.pi_.T.reshape(C * n,))
            Pi = np.vstack((np.diag(self.pi_[:, c_]) for c_ in xrange(C)))
            b = (D - Pi.dot(Pi.T)).dot(self.f_.T.reshape((C * n,)))
            b = b.reshape((n, C))
            b = b + y - self.pi_
            c = np.hstack((E[c_].dot(self.K_).dot(b[:, c_])[:, np.newaxis] for c_ in xrange(C)))
            _t = np.sum(c, axis=1)
            _t2 = solve_triangular(M, _t)
            _t3 = solve_triangular(M, _t2, trans='T')
            _t4 = np.hstack((E[c_].dot(_t3)[:, np.newaxis] for c_ in xrange(C)))
            a = b - c + _t4
            self.f_ = self.K_.dot(a)
            lml = -0.5 * sum(a[:, _c].dot(self.f_[:, _c]) for _c in xrange(C)) # -0.5a^Tf
            lml += sum(y[:, _c].dot(self.f_[:, _c]) for _c in xrange(C)) # y^Tf
            lml -= sum(log_sum_exp(f) for f in self.f_)
            lml -= sum(z)
            lmls.append(lml)
            if len(lmls) >= 2 and np.abs(lmls[-1] - lmls[-2]) < self.tol:
                break
        self.lml_ = lmls[-1]

    def _predict(self, X):
        pass

    def _serialize(self, params):
        return params

    def _deserialize(self, params):
        return params