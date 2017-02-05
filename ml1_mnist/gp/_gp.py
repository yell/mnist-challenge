import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse.linalg import cg
from scipy.sparse import diags as sparse_diag

import env
from base import BaseEstimator
from utils import RNG, one_hot, one_hot_decision_function
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
    Gaussian processes classifier (GPC).

    Parameters
    ----------
    kernel : {'rbf'}, optional
        Specifies the kernel type to be used in the algorithm.
        Currently only rbf is supported.
    kernel_params : dict, optional
        Initial params of the `kernel`.
    sigma_n : non-negative float
        Noise standard deviation.
    max_iter : positive int, optional
        Maximum number of Newton iterations.
    tol : positive float, optional
        Tolerance for approx. LML for Newton iterations.
    algorithm : {'exact', 'cg'}, optional
        Algorithm to solve the underlying linear systems.
    cg_tol : positive float, optional
        Tolerance for CG if `algorithm` is set to 'cg'.
    cg_max_iter : positive int, optional
        Maximum number of iterations for CG
        if `algorithm` is set to 'cg'.
    random_seed : None or int, optional
        Pseudo-random number generator seed used for random sampling.

    Attributes
    ----------
    K_ : (n_samples, n_samples) np.ndarray
        Covariance function.
    f_ : (n_samples, n_classes) np.ndarray
        Posterior approximation mode.
    lml_ : float
        Approx. log marginal likelihood \log{q(y|X, theta)},
        where theta are kernel parameters.
        Note that if `algorithm` is set to 'cg', lml_ is 2 * log|B|
        larger than the actual value (since the latter in this case
        is not computed).

    Examples
    --------
    >>> from utils import one_hot, one_hot_decision_function
    >>> from metrics import accuracy_score, log_loss
    >>> from nn.activations import softmax
    >>> X = [[0., 0.],
    ...      [0., 1.],
    ...      [1., 0.],
    ...      [1., 1.]]
    >>> y = np.array([0, 1, 1, 0])
    >>> gp = GPClassifier(random_seed=1337, kernel_params=dict(sigma=1., gamma=1.))
    >>> gp.fit(X, y).K_
    array([[ 1.        ,  0.36787945,  0.36787945,  0.13533528],
           [ 0.36787945,  1.        ,  0.13533528,  0.36787945],
           [ 0.36787945,  0.13533528,  1.        ,  0.36787945],
           [ 0.13533528,  0.36787945,  0.36787945,  1.        ]], dtype=float32)
    >>> pi = softmax(gp.f_); pi
    array([[ 0.57933426,  0.42066577],
           [ 0.42101488,  0.5789851 ],
           [ 0.42103583,  0.57896417],
           [ 0.5786863 ,  0.42131364]], dtype=float32)
    >>> y_pred = one_hot_decision_function(pi); y_pred
    array([[ 1.,  0.],
           [ 0.,  1.],
           [ 0.,  1.],
           [ 1.,  0.]], dtype=float32)
    >>> accuracy_score(one_hot(y), y_pred)
    1.0
    >>> log_loss(one_hot(y), pi) #doctest: +ELLIPSIS
    0.546...
    >>> gp.lml_ #doctest: +ELLIPSIS
    -3.996...
    >>> X_star = [[0., 0.09], [0.3, 0.5], [-3., 4.]]
    >>> gp.predict_proba(X_star) # random
    array([[ 0.56200714,  0.43799286],
           [ 0.4980865 ,  0.5019135 ],
           [ 0.49546654,  0.50453346]])
    >>> gp.predict(X_star)
    array([[ 1.,  0.],
           [ 0.,  1.],
           [ 0.,  1.]])
    >>> gp.set_params(algorithm='cg').fit(X, y).predict_proba(X_star) # random
    array([[ 0.56027713,  0.43972287],
           [ 0.49784431,  0.50215569],
           [ 0.49546654,  0.50453346]])
    >>> gp.lml_ #doctest: +ELLIPSIS
    -2.439...

    >>> from utils.dataset import load_mnist
    >>> from model_selection import TrainTestSplitter as TTS
    >>> X, y = load_mnist('train', '../../data/')
    >>> train, _ = TTS(random_seed=1337, shuffle=True).split(y, train_ratio=0.0015, stratify=True)
    >>> X = X[train]; X.shape
    (84, 784)
    >>> y = one_hot(y[train])
    >>> X /= 255.
    >>> gp = GPClassifier(random_seed=1337, kernel_params=dict(sigma=1., gamma=1.))
    >>> pi = softmax(gp.fit(X, y).f_);
    >>> accuracy_score(y, one_hot_decision_function(pi))
    1.0
    >>> log_loss(y, pi) #doctest: +ELLIPSIS
    1.571...
    >>> gp.lml_ #doctest: +ELLIPSIS
    -199.66...
    """

    def __init__(self, kernel='rbf', kernel_params={}, sigma_n=0.0,
                 max_iter=100, tol=1e-5, algorithm='exact', cg_tol=1e-5, cg_max_iter=None,
                 n_samples=1000, random_seed=None):
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.sigma_n = sigma_n
        self.max_iter = max_iter
        self.tol = tol
        self.algorithm = algorithm
        self.cg_tol = cg_tol
        self.cg_max_iter = cg_max_iter
        self.n_samples = n_samples
        self.random_seed = random_seed

        self.K_ = None
        self.f_ = None
        self.pi_ = None
        self.lml_ = None

        self._kernel = None
        self._e = None
        self._M = None
        super(GPClassifier, self).__init__(_y_required=True)

    def _fit(self, X, y):
        """
        Compute mode of approximation of the posterior using
        algorithm (3.3) from GPML with shared covariance matrix
        among all latent functions.
        """
        if len(y.shape) == 1 or y.shape[1] == 1:
            y = one_hot(y)
        self._check_X_y(X, y)
        y = y.astype(np.float32)
        self._kernel = get_kernel(self.kernel, **self.kernel_params)
        # shortcuts
        C = self._n_outputs
        n = self._n_samples
        # construct covariance matrix [if needed]
        # if self.K_ is None:
        self.K_ = self._kernel(X, X)
        self.K_ += self.sigma_n**2 * np.eye(n)
        self.K_ = self.K_.astype(np.float32)

        # init latent function values
        self.f_ = np.zeros_like(y)

        lmls = []
        iter = 0
        while True:
            iter += 1
            if iter > self.max_iter:
                print 'convergence is not reached'
                return

            self.pi_ = softmax(self.f_)
            z = []
            self._e = []
            for c_ in xrange(C):
                # compute E_c
                sqrt_d_c = np.sqrt(self.pi_[:, c_])
                _T = np.eye(self._n_samples) + (sqrt_d_c * self.K_.T).T * sqrt_d_c
                if self.algorithm == 'exact':
                    L = cholesky(_T, lower=True, overwrite_a=True)
                    _T2 = solve_triangular(L, sqrt_d_c)
                    e_c = sqrt_d_c * solve_triangular(L, _T2, trans='T')
                elif self.algorithm == 'cg':
                    _t, _ = cg(_T, sqrt_d_c, tol=self.cg_tol, maxiter=self.cg_max_iter)
                    _t = _t.astype(np.float32)
                    e_c = sqrt_d_c * _t
                self._e.append(e_c)
                # compute z_c
                if self.algorithm == 'exact':
                    z_c = sum(np.log(L.diagonal()))
                    z.append(z_c)
            # compute b
            # b = (D - Pi.dot(Pi.T)).dot(self.f_.T.reshape((C * n,)))
            # b = b.reshape((n, C))
            b = (1. - self.pi_) * self.pi_ * self.f_
            b = b + y - self.pi_
            # compute c
            c = np.hstack((self._e[c_] * self.K_.dot(b[:, c_]))[:, np.newaxis] for c_ in xrange(C))
            # compute a
            # self._M = cholesky(np.diag(sum(self._e)), lower=True, overwrite_a=True)
            # _t = np.sum(c, axis=1)
            # _t2 = solve_triangular(self._M, _t)
            # _t3 = solve_triangular(self._M, _t2, trans='T')
            _t3 = np.sum(c, axis=1) / np.maximum(sum(self._e), 1e-8 * np.ones_like(self._e[0]))
            _t4 = np.hstack((self._e[c_] * _t3)[:, np.newaxis] for c_ in xrange(C))
            a = b - c + _t4
            a = a.astype(np.float32)
            # compute f
            self.f_ = self.K_.dot(a)
            # compute approx. LML
            lml = -0.5 * sum(a[:, _c].dot(self.f_[:, _c]) for _c in xrange(C)) # -0.5a^Tf
            lml += sum(y[:, _c].dot(self.f_[:, _c]) for _c in xrange(C)) # y^Tf
            lml -= sum(log_sum_exp(f) for f in self.f_)
            lml -= sum(z)
            lmls.append(lml)
            if len(lmls) >= 2 and np.abs(lmls[-1] - lmls[-2]) < self.tol * self.K_.max():
                break
        self.lml_ = lmls[-1]

    def _predict_k_star(self, k_star, x_star):
        """
        Predict one test sample using algorithm (3.4) from GPML
        and assuming shared covariance matrix among all latent functions.
        """
        # shortcuts
        C = self._n_outputs
        n = self._n_samples

        mu = (self._y - self.pi_).T.dot(k_star)
        Sigma = []
        k_star_star = self._kernel(x_star, x_star)
        for c_ in xrange(C):
            b = self._e[c_] * k_star
            # _t = solve_triangular(self._M, b)
            # _t2 = solve_triangular(self._M, _t, trans='T')
            _t2 = b / np.maximum(sum(self._e), 1e-8 * np.ones_like(self._e[0]))
            c = self._e[c_] * _t2
            sigma_row = [c.dot(k_star)] * C
            sigma_row[c_] += ( k_star_star - b.dot(k_star) )
            Sigma.append(sigma_row)
        Sigma = np.asarray(Sigma)
        f_star = self._rng.multivariate_normal(mu, Sigma, size=self.n_samples)
        pi_star = softmax(f_star)
        return np.mean(pi_star, axis=0)

    def predict_proba(self, X):
        self._kernel = get_kernel(self.kernel, **self.kernel_params)
        K_star = self._kernel(X, self._X)
        self._rng = RNG(self.random_seed)
        predictions = [self._predict_k_star(k_star, x_star) for k_star, x_star in zip(K_star, X)]
        return np.asarray(predictions)

    def _predict(self, X):
        pi_pred = self.predict_proba(X)
        return one_hot_decision_function(pi_pred)

    def reset_K(self):
        self.K_ = None

    def _serialize(self, params):
        for attr in ('K_', 'f_', 'pi_'):
            if attr in params and params[attr] is not None:
                params[attr] = params[attr].tolist()
        return params
        
    def _deserialize(self, params):
        for attr in ('K_', 'f_', 'pi_'):
            if attr in params and params[attr] is not None:
                params[attr] = np.asarray(params[attr])
        return params


# if __name__ == '__main__':
#     # run corresponding tests
#     from utils.testing import run_tests
#     run_tests(__file__)

if __name__ == '__main__':
    from utils.dataset import load_mnist
    from utils import Stopwatch, one_hot, one_hot_decision_function
    from utils.read_write import load_model
    from model_selection import TrainTestSplitter
    from metrics import accuracy_score

    X, y = load_mnist('train', '../../data/')
    train, test = TrainTestSplitter(random_seed=1337, shuffle=True).split(y, train_ratio=0.0015, stratify=True)
    X /= 255.

    gp = GPClassifier(max_iter=100,
                      tol=1e-6,
                      random_seed=1337,
                      sigma_n=0.1,
                      algorithm='cg',
                      kernel_params=dict(
                          gamma=0.08)
                      )
    with Stopwatch(verbose=True): # Elapsed time: 0.741 sec
        gp.fit(X[train], one_hot(y[train]))
    # print gp.predict_proba(X[test][:5])
    # print one_hot(y[test][:5])
    print gp.evaluate(X[test][:25], one_hot(y[test][:25]))
    print gp.lml_

#     X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
#     y = one_hot([0, 1, 1, 0])
#     gp = GPClassifier(
#         algorithm='cg',
#         random_seed=1337,
#         kernel_params=dict(sigma=1., gamma=1.)
#     )
#     with Stopwatch(verbose=True): # Elapsed time: 0.002 sec
#         gp.fit(X, y)
#     print gp.lml_
#     print gp.predict_proba([[0.3, 0.5], [0., 0.09], [-3., 4.]])
#     gp.save('gp.json', json_params=dict(indent=4))
#     gp_loaded = load_model('gp.json').fit(X, y)
#     print gp_loaded.predict_proba([[0.3, 0.5], [0., 0.09], [-3., 4.]])
#     # print gp._e
#     # print np.maximum(sum(gp._e), 0.52 * np.ones_like(gp._e[0]))

#     # # -3.99583767262
#     # # [[0.50210975  0.49789025]
#     # #  [0.55729945  0.44270055]
#     # #  [0.49546654  0.50453346]]