import numpy as np

import env
from base import BaseEstimator
from layers import FullyConnected, Activation
from metrics import get_metric
from optimizers import get_optimizer
from activations import get_activation
from utils import RNG, one_hot_decision_function
from model_selection import TrainTestSplitter


class NNClassifier(BaseEstimator):
    def __init__(self, layers, n_batches=10,
                 loss='categorical_crossentropy', metric='accuracy_score',
                 optimizer='adam', optimizer_params={},
                 shuffle=True, random_seed=None):
        self.layers = layers
        self._n_layers = len(self.layers)
        self.n_batches = n_batches # mini-batches will be generated in the stratified manner
        self.loss = loss
        if self.loss == 'categorical_crossentropy':
            self._loss_grad = lambda actual, predicted: -(actual - predicted)
        self._loss = get_metric(self.loss)
        self.metric = metric
        self._metric = get_metric(self.metric)
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self._optimizer = get_optimizer(self.optimizer, **self.optimizer_params)
        self._initialized = False
        self._training = False
        self.shuffle = shuffle
        self.random_seed = random_seed
        for layer in self.layers:
            layer.random_seed = self.random_seed
        self._tts = TrainTestSplitter(shuffle=True, random_seed=random_seed)
        super(NNClassifier, self).__init__(_y_required=True) # TODO: split into multiple NNs later

    def _setup_layers(self, X_shape):
        for layer in self.layers:
            layer.setup_weights(X_shape) # allocate and initialize
            X_shape = layer.shape(prev_shape=X_shape) # forward propagate shape
        self._initialized = True
        print "Total number of parameters: ", self.n_params

    def forward_pass(self, X_batch):
        Z = X_batch
        for layer in self.layers:
            Z = layer.forward_pass(Z)
        return Z

    def parametric_layers(self):
        for layer in self.layers:
            if hasattr(layer, 'W'):  # TODO: more generic solution
                yield layer

    def batch_iter(self):
        for indices in self._tts.make_k_folds(self._y, n_folds=self.n_batches, stratify=True):
            yield self._X[indices], self._y[indices]

    def update(self, X_batch, y_batch):
        # forward pass
        y_pred = self.forward_pass(X_batch)

        # backward pass
        grad = self._loss_grad(y_batch, y_pred)
        for layer in reversed(self.layers):
            grad = layer.backward_pass(grad)
        return self._loss(y_batch, y_pred)

    def _fit(self, X, y, X_val=None, y_val=None):
        if y.ndim == 1:
            y = y[:, np.newaxis]
        if not self._initialized:
            self._setup_layers(X.shape)
        self._X_val = X_val
        self._y_val = y_val
        print "Train on {0} samples, validate on {1} samples\n".format(len(X), len(X_val))
        self._training = True
        self._optimizer.optimize(self)
        self._training = False

    def validate_proba(self, X=None): # can be called during training
        if self._training:
            self._training = False
        if X is None:
            y_pred = self.forward_pass(self._X)
        else:
            y_pred = self.forward_pass(X)
        self._training = True
        return y_pred

    def validate(self, X=None):
        y_pred = self.validate_proba(X)
        return one_hot_decision_function(y_pred)

    def predict_proba(self, X):
        y_pred = self.forward_pass(self._X)
        return y_pred

    def predict(self, X):
        y_pred = self.predict_proba()
        return one_hot_decision_function(y_pred)

    def _serialize(self, params):
        return params

    def _deserialize(self, params):
        return params

    @property
    def n_params(self):
        return sum(layer.n_params for layer in self.layers)



if __name__ == '__main__':
    nn = NNClassifier(layers=[
        # FullyConnected(2500),
        # Activation('leaky_relu'),
        # FullyConnected(2000),
        # Activation('leaky_relu'),
        # FullyConnected(1000),
        # Activation('softmax'),
        FullyConnected(32),
        Activation('leaky_relu'),
        FullyConnected(10),
        Activation('softmax')
    ], n_batches=30, random_seed=1337, optimizer_params=dict(max_epochs=100, verbose=True))
    from utils.dataset import load_mnist
    from utils import one_hot
    X, y = load_mnist(mode='train', path='../../data/')
    X /= 255.
    train, test = TrainTestSplitter(shuffle=True, random_seed=1337).split(y, train_ratio=0.85)
    y = one_hot(y)
    nn.fit(X[train], y[train], X_val=X[test], y_val=y[test])