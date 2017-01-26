import numpy as np
from collections import defaultdict

from utils import (print_inline, width_format,
                   Stopwatch, plot_learning_curves)


def get_optimizer(optimizer_name, **params):
    for k, v in globals().items():
        if k.lower() == optimizer_name.lower():
            return v(**params)
    raise ValueError("invalid optimizer name '{0}'".format(optimizer_name))


class BaseOptimizer(object):
    def __init__(self, max_epochs=100, verbose=False):
        self.max_epochs = max_epochs
        self.verbose = verbose

    def _setup(self, nnet):
        pass

    def update(self, nnet):
        raise NotImplementedError()

    def train_epoch(self, nnet):
        self._setup(nnet)
        losses = []
        for X_batch, y_batch in nnet.batch_iter():
            if self.verbose: print_inline('.')
            loss = np.mean(nnet.update(X_batch, y_batch))
            self.update(nnet)
            losses.append(loss)
        if self.verbose: print
        return losses # epoch losses

    def optimize(self, nnet):
        loss_history = []
        score_history = []
        val_loss_history = []
        val_score_history = []
        timer = Stopwatch(verbose=False).start()
        for i in xrange(self.max_epochs):
            if self.verbose:
                print_inline('Epoch {0:>{1}}/{2} '.format(i + 1, len(str(self.max_epochs)), self.max_epochs))
            losses = self.train_epoch(nnet)
            loss_history.append(losses)
            msg = 'elapsed: {0} sec'.format(width_format(timer.elapsed(), default_width=5, max_precision=2))
            msg += ' - loss: {0}'.format(width_format(np.mean(losses), default_width=5, max_precision=4))
            score = nnet._metric(nnet._y, nnet.validate())
            score_history.append(score)
            # TODO: change acc to metric name
            msg += ' - acc.: {0}'.format(width_format(score, default_width=6, max_precision=4))
            if nnet._X_val is not None:
                val_loss = nnet._loss(nnet._y_val, nnet.validate_proba(nnet._X_val))
                val_loss_history.append(val_loss)
                val_score = nnet._metric(nnet._y_val, nnet.validate(nnet._X_val))
                val_score_history.append(val_score)
                msg += ' - val. loss: {0}'.format(width_format(val_loss, default_width=5, max_precision=4))
                # TODO: fix acc.
                msg += ' - val. acc.: {0}'.format(width_format(val_score, default_width=6, max_precision=4))
            if self.verbose: print msg
            if i > 0:
                plot_learning_curves(loss_history, score_history, val_loss_history, val_score_history)
        return loss_history, score_history, val_loss_history, val_score_history


class Adam(BaseOptimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, **params):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 1
        super(Adam, self).__init__(**params)

    def _setup(self, nnet):
        # accumulators for moment and velocity
        self.ms, self.vs = defaultdict(dict), defaultdict(dict)
        for i, layer in enumerate(nnet.parametric_layers()):
            self.ms[i]['W'] = np.zeros_like(layer.W)
            self.ms[i]['b'] = np.zeros_like(layer.b)
            self.vs[i]['W'] = np.zeros_like(layer.W)
            self.vs[i]['b'] = np.zeros_like(layer.b)

    def update(self, nnet):
        for i, layer in enumerate(nnet.parametric_layers()):
            self.ms[i]['W'] = self.beta_1 * self.ms[i]['W'] + (1. - self.beta_1) * layer.dW
            self.ms[i]['b'] = self.beta_1 * self.ms[i]['b'] + (1. - self.beta_1) * layer.db
            self.vs[i]['W'] = self.beta_2 * self.vs[i]['W'] + (1. - self.beta_2) * layer.dW ** 2
            self.vs[i]['b'] = self.beta_2 * self.vs[i]['b'] + (1. - self.beta_2) * layer.db ** 2
            lr = self.learning_rate * np.sqrt(1. - self.beta_2 ** self.t) / (1. - self.beta_1 ** self.t)
            W_step = lr * self.ms[i]['W'] / (np.sqrt(self.vs[i]['W']) + self.epsilon)
            b_step = lr * self.ms[i]['b'] / (np.sqrt(self.vs[i]['b']) + self.epsilon)
            layer.W -= W_step
            layer.b -= b_step
        self.t += 1