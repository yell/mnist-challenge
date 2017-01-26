import numpy as np

import env
from base import BaseEstimator
from nn import NNClassifier
from nn.layers import FullyConnected, Activation


class LogisticRegression(BaseEstimator):
    """Multinomial Logistic Regression.

    Parameters
    ----------
    L1, L2 : non-negative float
        Regularization parameters
    n_batches : int
        Number of batches.
    optimizer : {'adam'}, optional
        Specifies which optimizer to use in the algorithm.
    optimizer_params : kwargs, optional
        Additional kwargs passed to `optimizer`
    shuffle : bool, optional
        Whether to shuffle the dataset.
    random_seed : int or None, optional
        Pseudo-random number generator seed used for random sampling.
    """
    def __init__(self, L1=0.0, L2=0.0, n_batches=10,
                 optimizer='adam', optimizer_params={},
                 shuffle=True, random_seed=None):
        self.L1 = L1
        self.L2 = L2
        self.n_batches = n_batches
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.shuffle = shuffle
        self.random_seed = random_seed
        self._nnet = None
        super(LogisticRegression, self).__init__(_y_required=True)

    def _fit(self, X, y, X_val=None, y_val=None):
        if self._nnet is None:
            self._nnet = NNClassifier(
                layers=[
                    FullyConnected(y.shape[1], L1=self.L1, L2=self.L2),
                    Activation('softmax')
                ],
                loss='categorical_crossentropy',
                metric='accuracy_score',
                n_batches=self.n_batches,
                optimizer=self.optimizer,
                optimizer_params=self.optimizer_params,
                shuffle=self.shuffle,
                random_seed=self.random_seed
            )
        self._nnet.fit(X, y, X_val=X_val, y_val=y_val)

    def _predict(self, X):
        return self._nnet.predict(X)

    def _serialize(self, params):
        return params

    def _deserialize(self, params):
        return params


if __name__ == '__main__':
    import env
    from utils.dataset import load_mnist
    from utils import one_hot
    X, y = load_mnist(mode='train', path='../../data/')
    X /= 255.
    logreg = LogisticRegression(n_batches=10,
                                random_seed=1337,
                                optimizer_params=dict(
                                    max_epochs=10,
                                    learning_rate=1e-3,
                                    plot=False)
                                )
    y = one_hot(y)
    logreg.fit(X[:1000], y[:1000], X_val=X[1000:2000], y_val=y[1000:2000])
    logreg.fit(X[:1000], y[:1000], X_val=X[1000:2000], y_val=y[1000:2000])
    y_pred = logreg.predict(X[1000:2000])
    print logreg._nnet._metric(y_pred, y[1000:2000])