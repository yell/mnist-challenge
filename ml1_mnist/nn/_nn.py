import numpy as np

import env
from base import BaseEstimator
from layers import FullyConnected, Activation
from metrics import get_metric
from optimizers import get_optimizer
from activations import get_activation
from utils import RNG


class NeuralNetwork(BaseEstimator):
    def __init__(self, layers, batch_size=128,
                 loss='categorical_crossentropy', metric='accuracy_score',
                 optimizer='adam', optimizer_params={},
                 shuffle=True, random_seed=None):
        self.layers = layers
        self._n_layers = len(self.layers)
        self.batch_size = batch_size
        self.loss = loss
        if self.loss == 'categorical_crossentropy':
            self._loss_grad = lambda actual, predicted: -(actual - predicted)
        self._loss = get_metric(self.loss)
        self.metric = metric
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self._optimizer = get_optimizer(self.optimizer, **self.optimizer_params)
        self._initialized = False
        self._training = False
        super(NeuralNetwork, self).__init__(_y_required=True) # TODO: split into multiple NNs later

    def _setup_layers(self, X_shape):
        x_shape = [self.batch_size] + list(X_shape[1:])
        for layer in self.layers:
            layer.setup_weights(x_shape) # allocate and initialize
            x_shape = layer.shape(prev_shape=x_shape) # forward propagate shape
        self._initialized = True
        print "Total number of parameters: ", self.n_params

    def forward_pass(self, X_batch):
        Z = X_batch
        for layer in self.layers:
            Z = layer.forward_pass(Z)
        return Z

    def update(self, X_batch, y_batch):
        # forward pass
        y_pred = self.forward_pass(X_batch)

        # backward pass
        grad = self._loss_grad(y_batch, y_pred)
        for layer in reversed(self.layers):
            grad = layer.backward_pass(grad)
        return self._loss(y_batch, y_pred)


    def _fit(self, X, y):
        if y.ndim == 1:
            y = y[:, np.newaxis]
        if not self._initialized:
            self._setup_layers(X.shape)
        self._training = True
        # TODO: pass to optimizer
        self._training = False

    def _predict(self, X):
        pass

    def _serialize(self, params):
        return params

    def _deserialize(self, params):
        return params

    @property
    def n_params(self):
        return sum(layer.n_params for layer in self.layers)


if __name__ == '__main__':
    nn = NeuralNetwork(layers=[
        FullyConnected(10),
        Activation('leaky_relu'),
        FullyConnected(2),
        Activation('softmax')
    ], batch_size=16)
    from utils.dataset import load_mnist
    X, y = load_mnist(mode='train', path='../../data/')
    nn.fit(X[:16], y[:16])
    # print nn.layers[0].W.shape
    # print nn.layers[0].b.shape
    # print nn.layers[2].W.shape
    # print nn.layers[2].b.shape
    # print nn.n_params
