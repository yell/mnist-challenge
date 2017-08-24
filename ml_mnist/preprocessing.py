import numpy as np


class StandardScaler(object):
    """
    Standardize features by removing the mean
    and scaling to unit variance.

    Parameters
    ----------
    copy : bool, optional
        If False, do inplace scaling.
    with_mean : bool, optional
        If True, center the data before scaling.
    with_std : bool, optional
        If True, scale the data to unit variance.

    Attributes
    ----------
    mean_ : (n_features,) np.ndarray
        The mean value for each feature.
    std_ : (n_features,) np.ndarray
        The standart deviation for each feature.

    Examples
    --------
    >>> X = (np.arange(12.) ** 2).reshape((4, 3))
    >>> X
    array([[   0.,    1.,    4.],
           [   9.,   16.,   25.],
           [  36.,   49.,   64.],
           [  81.,  100.,  121.]])
    >>> ss = StandardScaler().fit(X)
    >>> ss.mean_, ss.std_
    (array([ 31.5,  41.5,  53.5]), array([ 31.5       ,  37.9769667 ,  44.52246624]))
    >>> X_new = ss.transform(X)
    >>> X_new
    array([[-1.        , -1.06643588, -1.11179825],
           [-0.71428571, -0.67145963, -0.64012626],
           [ 0.14285714,  0.19748813,  0.23583599],
           [ 1.57142857,  1.54040739,  1.51608852]])
    >>> StandardScaler(with_std=False).fit_transform(X)
    array([[-31.5, -40.5, -49.5],
           [-22.5, -25.5, -28.5],
           [  4.5,   7.5,  10.5],
           [ 49.5,  58.5,  67.5]])
    """
    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.std_ = None
        self._called_fit = False

    def _check_X(self, X):
        if self.copy:
            _X = np.array(X)
        elif isinstance(X, np.ndarray):
            _X = X
        else:
            _X = np.asarray(X)
        return _X

    def fit(self, X):
        _X = self._check_X(X)
        if self.with_mean:
            self.mean_ = np.mean(_X, axis=0)
        if self.with_std:
            self.std_ = np.std(_X, axis=0)
        self._called_fit = True
        return self

    def transform(self, X):
        if not self._called_fit:
            raise ValueError('`fit` must be called before calling `predict`')
        X_new = self._check_X(X)
        if self.with_mean:
            X_new -= self.mean_
        if self.with_std:
            X_new /= self.std_
        return X_new

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


if __name__ == '__main__':
    # run corresponding tests
    from utils.testing import run_tests
    run_tests(__file__)
