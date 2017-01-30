import numpy as np

from ml1_mnist.nn import NNClassifier
from ml1_mnist.nn.layers import FullyConnected, Activation, Dropout
from ml1_mnist.model_selection import TrainTestSplitter
from ml1_mnist.augmentation import RandomAugmentator
from ml1_mnist.metrics import accuracy_score
from ml1_mnist.utils import one_hot
from ml1_mnist.utils.dataset import load_mnist
from ml1_mnist.utils.read_write import load_model

def run():
	i = 0
	for dropouts in (
		# [0.05, 0.1],
		# [0.05, 0.05],
		[0., 0.],
		# [0.1, 0.1],
		# [0., 0.1],
		# [0.1, 0.2],
	):
		i += 1
		# if i <= 2:
		# 	continue
		# print "Loading data ..."
		# X = np.load('data/X_aug_nn.npy')#[:30000]
		# y = np.load('data/y_aug_nn.npy')#[:30000]
		# train, test = TrainTestSplitter(shuffle=True, random_seed=1337).split(y, train_ratio=29./30.)

		print "Loading data ..."
		X, y = load_mnist(mode='train', path='data/')
		X = X / 255.
		X = X.astype(np.float32)

		tts = TrainTestSplitter(shuffle=False, random_seed=1337)
		train, val = tts.split(y, train_ratio=55005.98/60000., stratify=True) # 55k : 5k
		X_train, y_train, X_val, y_val = X[train], y[train], X[val], y[val]

		y_val = one_hot(y_val)

		aug = RandomAugmentator(transform_shape=(28, 28), random_seed=1337)
		aug.add('RandomRotate', angle=(-5., 7.))
		aug.add('RandomGaussian', sigma=(0., 0.5))
		aug.add('RandomShift', x_shift=(-1, 1), y_shift=(-1, 1))

		print "Augmenting data ..."
		X_train = aug.transform(X_train, 4)
		y_train = np.repeat(y_train, 5)
		y_train = one_hot(y_train)

		# TODO: L1, L2, maxnorm if still overfits
		# for dropout in (0.05, 0.1, 0.2, 0.5, 0.25):
		# print "dropout = {0:.2f}".format(dropout)
		nn = NNClassifier(layers=[
							  FullyConnected(800),
		                      Activation('leaky_relu'),
		                      Dropout(dropouts[0]),
		                      FullyConnected(1000),
		                      Activation('leaky_relu'),
		                      Dropout(dropouts[1]),
		                      FullyConnected(800),
		                      Activation('leaky_relu'),
		                      FullyConnected(500),
		                      Activation('leaky_relu'),
		                      # Dropout(0.1),
		                      FullyConnected(250),
		                      Activation('leaky_relu'),
		                      # Dropout(dropout),
		                      FullyConnected(10),
		                      Activation('softmax')
		                  ],
		                  # n_batches=128,
		                  n_batches=1024,
		                  # n_batches=10,
		                  shuffle=True,
		                  random_seed=1337,
		                  save_weights=False,
		                  optimizer_params=dict(
		                      # max_epochs=500,
		                      max_epochs=70,
		                      # max_epochs=25,
		                      early_stopping=12,
		                      verbose=True,
		                      plot=False,
		                      # plot_dirpath='tmp/learning_curves{0}/'.format(i),
		                      learning_rate=5e-5
		                      # learning_rate=1e-3
		                  ))
		    
		print "Initializing NN ..."
		nn.fit(X_train, y_train, X_val=X_val, y_val=y_val)
		acc = nn.evaluate(X_val, y_val, 'accuracy_score')
		print acc
		
		fname = 'nn.json'.format(i)
		nn.save(fname)


if __name__ == '__main__':
	run()