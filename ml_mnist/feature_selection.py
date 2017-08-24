import numpy as np


class VarianceThreshold(object):
    """Feature selector that removes all low-variance features.

    Parameters
    ----------
    threshold : float, optional
        Features with a standard deviation lower than `threshold` will be
        removed. The default is to keep all features with non-zero variance,
        i.e. remove the features that have the same value in all samples.

    Examples
    --------
    >>> X = [[0, 2, 0, 3],
    ...      [0, 1, 4, 3],
    ...      [0, 1, 1, 3]]
    >>> selector = VarianceThreshold(0.0)
    >>> selector.fit_transform(X)
    array([[2, 0],
           [1, 4],
           [1, 1]])
    """
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def _check_X(self, X):
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        return X

    def fit_transform(self, X):
        _X = self._check_X(X)
        return _X[:, (np.std(_X, axis=0) > self.threshold + 1e-8)]


if __name__ == '__main__':
    # run corresponding tests
    from utils.testing import run_tests
    run_tests(__file__)