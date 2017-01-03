import numpy as np


class BaseEstimator(object):
    """Base class for all estimators."""

    def __init__(self, y_required=True):
        self.y_required = y_required

    def _validate_input(self, X, y=None):
        """
        Ensure inputs are in the expected format.

        - convert `X` [and `y`] to `np.ndarray` if needed
        - ensure `X` [and `y`] are not empty

        Parameters
        ----------
        X : (n_samples, n_features) array-like
            Data (feature vectors).
        y : (n_samples,) or (n_samples, n_outputs) array-like
            Labels vector. By default is required, but may be omitted
            if `y_required` is False.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError('number of features must be > 0')

        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])

        # self.X = X

        if self.y_required:
            if y is None:
                raise ValueError('missed required argument `y`')

            if not isinstance(y, np.ndarray):
                self.y = np.array(y)

            if y.size == 0:
                raise ValueError('number of outputs must be > 0')

        # self.y = y

    def _fit(self, X, y=None):
        raise NotImplementedError()

    def fit(self, X, y=None):
        self._validate_input(X, y)
        self._fit(X, y)


    # def _predict(self):
    #     raise NotImplementedError()
    #
    # def predict(self):
    #     pass
    #
    #
    # def _save(self):
    #     raise NotImplementedError()
    #
    # def save(self, filename='model.json'):
    #     # store as must attributes / params as possible and --> dict --> json
    #     # add _save()
    #     # store to file
    #     pass
    #
    #
    # def clone(self):
    #     pass
    #
    # def get_params(self):
    #     pass
    #
    # def _pprint(self):
    #     pass