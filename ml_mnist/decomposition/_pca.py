import numpy as np
from scipy.linalg import svd

import env
from base import BaseEstimator


class PCA(BaseEstimator):
    """
    Principal component analysis (PCA).
    Linear dimensionality reduction using Singular Value Decomposition
    of the data to project it to a lower dimensional space.

    Parameters
    ----------
    n_components : int or None
        Number of components to keep.
        If `n_components` is not set all components are kept:
        ``n_components == min(n_samples, n_features)``
    whiten : bool, optional
        When True, the `components_` vectors are multiplied by
        the square root of `n_samples` and then divided by the
        singular values to ensure uncorrelated outputs with
        unit component-wise variances.
        Whitening will remove some information from the transformed
        signal (the relative variance scales of the components)
        but can sometime improve the predictive accuracy of the
        downstream estimators by making their data respect some
        hard-wired assumptions.

    Attributes
    ----------
    components_ : (n_components, n_features) np.ndarray
        Principal axes in feature space, representing the directions
        of maximum variance in the data. The components are sorted by
        `explained_variance_`.
    explained_variance_ : (n_components,) np.ndarray
        The amount of variance explained by each of the selected components.
    explained_variance_ratio_ : (n_components,) np.ndarray
        Percentage of variance explained by each of the selected components.
    mean_ : (n_features,)
        Per-feature empirical mean, estimated from the training set.

    Examples
    --------
    >>> X = [[0.1, 0.2, 0.31],
    ...      [0.2, 0.3, 0.56],
    ...      [0.3, 0.4, 0.75],
    ...      [0.4, 0.5, 0.98]]
    >>> pca = PCA(n_components=2).fit(X); pca.mean_
    array([ 0.25,  0.35,  0.65])
    >>> pca.components_
    array([[ 0.38188512,  0.38188512,  0.84162196],
           [ 0.59511659,  0.59511659, -0.54006711]])
    >>> pca.explained_variance_
    array([  8.56061948e-02,   4.38052411e-05])
    >>> pca.explained_variance_ratio_
    array([  9.99488555e-01,   5.11444730e-04])
    >>> Y = [[1, 2, 3],
    ...      [4, 5, 6],
    ...      [7, 8, 9]]
    >>> Z = pca.transform(Y)
    >>> Z # representation of Y in basis of X's principal components
    array([[  2.89433587,   0.15912211],
           [  7.71051244,   2.10962032],
           [ 12.52668901,   4.06011854]])
    >>> Z.dot(pca.components_) + pca.mean_ # reconstruction
    array([[ 1.45,  1.55,  3.  ],
           [ 4.45,  4.55,  6.  ],
           [ 7.45,  7.55,  9.  ]])
    >>> pca.set_params(n_components=1).transform(Y) # no need to recompute
    array([[  2.89433587],
           [  7.71051244],
           [ 12.52668901]])


    >>> Z_whitened = pca.set_params(n_components=2, whiten=True).transform(X)
    >>> Z_whitened
    array([[-1.36957295,  0.768724  ],
           [-0.38940597, -1.6477347 ],
           [ 0.41817098,  0.83174497],
           [ 1.34080795,  0.04726573]])
    >>> import numpy as np
    >>> np.cov(Z_whitened.T, ddof=0)
    array([[  1.00000000e+00,  -7.67243907e-15],
           [ -7.67243907e-15,   1.00000000e+00]])


    >>> pca
    PCA(,
      components_=array([[ 0.38189,  0.38189,  0.84162],
           [ 0.59512,  0.59512, -0.54007]]),
      explained_variance_=array([  8.56062e-02,   4.38052e-05]),
      explained_variance_ratio_=array([  9.99489e-01,   5.11445e-04]),
      mean_=array([ 0.25,  0.35,  0.65]), n_components=2, whiten=True)
    >>> pca.save('pca.json', json_params=dict(indent=4))
    >>> from utils.read_write import load_model
    >>> pca_loaded = load_model('pca.json')
    >>> pca_loaded # no need to refit
    PCA(,
      components_=array([[ 0.38189,  0.38189,  0.84162],
           [ 0.59512,  0.59512, -0.54007]]),
      explained_variance_=array([  8.56062e-02,   4.38052e-05]),
      explained_variance_ratio_=array([  9.99489e-01,   5.11445e-04]),
      mean_=array([ 0.25,  0.35,  0.65]), n_components=2, whiten=True)
    """
    def __init__(self, n_components=None, whiten=False):
        self.whiten = whiten
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        super(PCA, self).__init__(_y_required=False)

    def _decompose(self, X):
        # copy data by default
        X = np.copy(X)
        X -= self.mean_
        self.n_components = self.n_components or min(self._n_samples, self._n_features)

        _, s, Vh = svd(X,
                       full_matrices=False,
                       compute_uv=True,
                       overwrite_a=True,
                       check_finite=False)
        self.components_ = Vh[:self.n_components]
        explained_variance = (s ** 2) / float(self._n_samples)
        self.explained_variance_ = explained_variance[:self.n_components]
        total_variance = sum(explained_variance)
        self.explained_variance_ratio_ = explained_variance[:self.n_components] / total_variance

    def _fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        if self._called_fit or self.components_ is None:
            self._decompose(X)

    def transform(self, X):
        self.n_components = self.n_components or min(self._n_samples, self._n_features)
        X = np.array(X, dtype=np.float)
        X -= self.mean_
        Z = np.dot(X, self.components_[:self.n_components].T)
        if self.whiten:
            Z /= np.sqrt(self.explained_variance_[:self.n_components])
        return Z

    def _serialize(self, params):
        for attr in ('components_',
                      'mean_',
                      'explained_variance_',
                      'explained_variance_ratio_'):
            if attr in params and params[attr] is not None:
                params[attr] = params[attr].tolist()
        return params

    def _deserialize(self, params):
        for attr in ('components_',
                     'mean_',
                     'explained_variance_',
                     'explained_variance_ratio_'):
            if attr in params and params[attr] is not None:
                params[attr] = np.asarray(params[attr], dtype=np.float)
        return params


if __name__ == '__main__':
    # run corresponding tests
    from utils.testing import run_tests
    run_tests(__file__)
