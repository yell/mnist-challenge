import numpy as np
from copy import deepcopy

import env
from base import BaseEstimator
from utils import RNG
from layers import FullyConnected
from activations import sigmoid


class RBM(BaseEstimator):
    def __init__(self, n_hidden=256, persistent=True, k=1,
                 batch_size=128, n_iter=10, learning_rate=0.1, momentum=0.9, verbose=False,
                 random_seed=None):
        self.n_hidden = n_hidden
        self.persistent = persistent
        self.k = k # k in CD-k / PCD-k
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.random_seed = random_seed

        self.W = None
        self.vb = None  # visible units bias
        self.hb = None # hidden units bias

        self.best_W = None
        self.best_vb = None
        self.best_hb = None
        self.best_epoch = None
        self.best_recon = np.inf

        self._dW = None
        self._dvb = None
        self._dhb = None

        self._rng = None
        self._persistent = None
        self._initialized = False
        super(RBM, self).__init__(_y_required=False)

    def propup(self, v):
        """Propagate visible units activation upwards to the hidden units."""
        z = np.dot(v, self.W) + self.hb
        return sigmoid(z)

    def sample_h_given_v(self, v0_sample):
        """Infer state of hidden units given visible units."""
        h1_mean = self.propup(v0_sample)
        h1_sample = self._rng.binomial(size=h1_mean.shape, n=1, p=h1_mean)
        return h1_mean, h1_sample

    def propdown(self, h):
        """Propagate hidden units activation downwards to the visible units."""
        z = np.dot(h, self.W.T) + self.vb
        return sigmoid(z)

    def sample_v_given_h(self, h0_sample):
        """Infer state of visible units given hidden units."""
        v1_mean = self.propdown(h0_sample)
        v1_sample = self._rng.binomial(size=v1_mean.shape, n=1, p=v1_mean)
        return v1_mean, v1_sample

    def gibbs_hvh(self, h0_sample):
        """Performs a step of Gibbs sampling starting from the hidden units."""
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return v1_mean, v1_sample, h1_mean, h1_sample

    def gibbs_vhv(self, v0_sample):
        """Performs a step of Gibbs sampling starting from the visible units."""
        raise NotImplementedError()

    def free_energy(self, v_sample):
        """Function to compute the free energy."""
        raise NotImplementedError()

    def update(self, X_batch):
        # compute positive phase
        ph_mean, ph_sample = self.sample_h_given_v(X_batch)

        # decide how to initialize chain
        if self._persistent is not None:
            chain_start = self._persistent
        else:
            chain_start = ph_sample

        # gibbs sampling
        for step in xrange(self.k):
            nv_means, nv_samples, \
            nh_means, nh_samples = self.gibbs_hvh(chain_start if step == 0 else nh_samples)

        # update weights
        self._dW  = self.momentum * self._dW + \
                    np.dot(X_batch.T, ph_mean) - np.dot(nv_samples.T, nh_means)
        self._dvb = self.momentum * self._dvb +\
                    np.mean(X_batch - nv_samples, axis=0)
        self._dhb = self.momentum * self._dhb +\
                    np.mean(ph_mean - nh_means, axis=0)
        self.W  += self.learning_rate * self._dW
        self.vb += self.learning_rate * self._dvb
        self.hb += self.learning_rate * self._dhb

        # remember state if needed
        if self.persistent:
            self._persistent = nh_samples

        return np.mean(np.square(X_batch - nv_means))

    def batch_iter(self, X):
        n_batches = len(X) / self.batch_size
        for i in xrange(n_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            X_batch = X[start:end]
            yield X_batch
        if n_batches * self.batch_size < len(X):
            yield X[end:]

    def train_epoch(self, X):
        mean_recons = []
        for X_batch in self.batch_iter(X):
            mean_recons.append(self.update(X_batch))
        return np.mean(mean_recons)

    def _fit(self, X):
        if not self._initialized:
            layer = FullyConnected(self.n_hidden,
                                   bias=0.,
                                   random_seed=self.random_seed)
            layer.setup_weights(X.shape)
            self.W = layer.W
            self.vb = np.zeros(X.shape[1])
            self.hb = layer.b
            self._dW = np.zeros_like(self.W)
            self._dvb = np.zeros_like(self.vb)
            self._dhb = np.zeros_like(self.hb)
            self._rng = RNG(self.random_seed)
        self._rng.reseed()
        for i in xrange(self.n_iter):
            mean_recon = self.train_epoch(X)
            if mean_recon < self.best_recon:
                self.best_recon = mean_recon
                self.best_epoch = i + 1
                self.best_W = self.W.copy()
                self.best_vb = self.vb.copy()
                self.best_hb = self.hb.copy()

    def _serialize(self, params):
        for attr in ('W', 'best_W',
                     'vb', 'best_vb',
                     'hb', 'best_hb'):
            if attr in params and params[attr] is not None:
                params[attr] = params[attr].tolist()
        return params

    def _deserialize(self, params):
        for attr in ('W', 'best_W',
                     'vb', 'best_vb',
                     'hb', 'best_hb'):
            if attr in params and params[attr] is not None:
                params[attr] = np.asarray(params[attr])
        return params


if __name__ == '__main__':
    X = RNG(seed=1337).rand(32, 64)
    rbm = RBM(n_hidden=100,
              k=4,
              batch_size=16,
              n_iter = 10,
              learning_rate=0.01,
              momentum=0.99,
              random_seed=1337)
    rbm.fit(X)
    # rbm.save('rbm.json', json_params=dict(indent=4))
    # from utils.read_write import load_model
    # rbm_loaded = load_model('rbm.json')
    # print rbm_loaded.best_W.shape