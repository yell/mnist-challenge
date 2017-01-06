from copy import deepcopy
import numpy as np

import env; from utils import import_trace


class BaseEstimator(object):
    """Base class for all estimators."""

    def __init__(self, _y_required=True):
        self._y_required = _y_required
        self._X = None
        self._y = None
        self._n_samples = None
        self._n_features = None
        self._n_outputs = None
        self._called_fit = False

    def _check_X_y(self, X, y=None):
        """
        Ensure inputs are in the expected format:

        Convert `X` [and `y`] to `np.ndarray` if needed
        and ensure `X` [and `y`] are not empty.

        Parameters
        ----------
        X : (n_samples, n_features) array-like
            Data (feature vectors).
        y : (n_samples,) or (n_samples, n_outputs) array-like
            Labels vector. By default is required, but may be omitted
            if `_y_required` is False.
        """
        # validate `X`
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError('number of features must be > 0')

        if X.ndim == 1:
            self._n_samples, self._n_features = 1, X.shape
        else:
            self._n_samples, self._n_features = X.shape[0], np.prod(X.shape[1:])

        # validate `y` if needed
        if self._y_required:
            if y is None:
                raise ValueError('missed required argument `y`')

            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if y.size == 0:
                raise ValueError('number of outputs must be > 0')

            # TODO: decide whether to check len(y) == self._n_samples (semi-supervised learning)?
            if y.ndim == 1:
                self._n_outputs = 1
            else:
                self._n_outputs = np.prod(y.shape[1:])

        self._X = X
        self._y = y

    def _fit(self, X, y=None, **params):
        """Class-specific `fit` routine."""
        raise NotImplementedError()

    def fit(self, X, y=None, **params):
        """Fit the model according to the given training data (infrastructure)."""
        self._check_X_y(X, y)
        self._fit(X, y, **params)
        self._called_fit = True

    def _predict(self, X=None, **params):
        """Class-specific `predict` routine."""
        raise NotImplementedError()

    def predict(self, X=None, **params):
        """Predict the target for the provided data (infrastructure)."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self._called_fit:
            return self._predict(X, **params)
        else:
            raise ValueError('`fit` must be called before calling `predict`')

    def get_params(self, deep=True):
        """Obtain parameters of the model.

        # omit "interesting" members
        # add model name (including import trace)

        Parameters
        ----------
        deep : bool, optional
            Whether to deepcopy all the attributes.

        Returns
        -------
        params : dict
            Parameters of the model. Includes all attributes (members,
            not methods), that not start with underscore ("_") and also model
            name being class name stored in 'model' parameter.
        """
        params = vars(self)
        params = {key: params[key] for key in params if not key.startswith('_')}
        trace = import_trace(
            module_path=__file__,
            main_package_name='ml1_mnist',
            include_main_package=False
        )
        class_name = self.__class__.__name__
        params['model'] = '.'.join([trace, class_name])
        if deep:
            params = deepcopy(params)
        return params

    def set_params(self, **params):
        """Set parameters of the model.

        Parameters
        ----------
        params : dict
            New parameters.

        Returns
        -------
        self
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def _load(self):
        """Load (additional) class-specific parameters."""
        raise NotImplementedError()

    def _save(self):
        """Save (additional) class-specific parameters."""
        raise NotImplementedError()

    # def save(self, filename='model.json'):
    #     # store as must attributes / params as possible and --> dict --> json
    #     # add _save()
    #     # store to file
    #     pass