# LOOK AT THE VERY BOTTOM OF THIS FILE

import numpy as np

from ml1_mnist.nn import NNClassifier, RBM
from ml1_mnist.nn.layers import FullyConnected, Activation, Dropout
from ml1_mnist.nn.activations import leaky_relu
from ml1_mnist.knn import KNNClassifier
from ml1_mnist.logreg import LogisticRegression
from ml1_mnist.decomposition import PCA

from ml1_mnist.utils import (Stopwatch, 
	                         print_inline,
	                         one_hot)
from ml1_mnist.utils.dataset import load_mnist
from ml1_mnist.utils.read_write import load_model
from ml1_mnist.metrics import accuracy_score
from ml1_mnist.augmentation import RandomAugmentator
from ml1_mnist.model_selection import TrainTestSplitter



def _train_nn(X, y):
	"""
	Parameters
	----------
	X : training data
	y : training labels (not one-hot encoded!)
	"""
	print "Running '_train_nn'"

	aug = RandomAugmentator(transform_shape=(28, 28), random_seed=1337)
	aug.add('RandomRotate', angle=(-5., 7.))
	aug.add('RandomGaussian', sigma=(0., 0.5))
	aug.add('RandomShift', x_shift=(-1, 1), y_shift=(-1, 1))
	
	print_inline("Applying augmentation ... ")
	with Stopwatch(verbose=True):
		X = X.astype(np.float32)
		X = aug.transform(X, 4)
		y = np.repeat(y, 5)

	train, test = TrainTestSplitter(shuffle=True, random_seed=1337).split(y, train_ratio=29./30.)
	X_train = X[train]
	y_train = y[train]
	y_train = one_hot(y_train)
	X_val = X[test]
	y_val = y[test]
	y_val = one_hot(y_val)

	nn = NNClassifier(layers=[
					      FullyConnected(1337),
					      Activation('leaky_relu'),
					      Dropout(0.05),
					      FullyConnected(911),
					      Activation('leaky_relu'),
					      Dropout(0.1),
					      FullyConnected(666),
					      Activation('leaky_relu'),
					      Dropout(0.),
					      FullyConnected(333),
					      Activation('leaky_relu'),
					      Dropout(0.),
					      FullyConnected(128),
					      Activation('leaky_relu'),
					      Dropout(0.),
					      FullyConnected(10),
					      Activation('softmax')
		              ],
	                  n_batches=1024,
	                  shuffle=True,
	                  random_seed=1337,
	                  optimizer_params=dict(
	                      max_epochs=42,
	                      # early_stopping=12,
	                      verbose=True,
	                      plot=False,
	                      save_weights=False,
	                      # plot_dirpath='tmp/learning_curves{0}/'.format(i),
	                      learning_rate=5e-5
	                  ))

	print "Training NN ..."
	nn.fit(X_train, y_train, X_val=X_val, y_val=y_val)

	print "Left '_train_nn'"
	return nn



def knn(load_nn=True):
	"""
	Output (if `load_nn` is True)
	-----------------------------
	Running 'knn'
	Loading data ...
	Loading NN ...
	Extracting feature vectors ...
	Elapsed time: 15.606 sec
	Building k-d tree ... Elapsed time: 0.216 sec
	Evaluating k-NN ... Elapsed time: 33.015 sec

	Test accuracy 0.9887 (error 1.13%)
	"""
	print "Running 'knn'"
	print "Loading data ..."
	X_train, y_train = load_mnist(mode='train', path='data/')
	X_test, y_test = load_mnist(mode='test', path='data/')
	X_train /= 255.
	X_test /= 255.

	if load_nn:
		print "Loading NN ..."
		nn = load_model('models/nn.json')
	else:
		nn = _train_nn(X_train.copy(), y_train.copy())

	print_inline("Extracting feature vectors ... ")
	with Stopwatch(verbose=True):
		nn.forward_pass(X_train)
		X_train = leaky_relu(nn.layers[13]._last_input)
		nn.forward_pass(X_test)
		X_test = leaky_relu(nn.layers[13]._last_input)

	knn = KNNClassifier(algorithm='kd_tree', k=3, p=2, weights='uniform')
	print_inline("Building k-d tree ... ")
	with Stopwatch(verbose=True): 
		knn.fit(X_train, y_train)

	print_inline("Evaluating k-NN ... ")
	with Stopwatch(verbose=True): 
		y_pred = knn.predict(X_test)
		acc = accuracy_score(y_test, y_pred)

	print "\nTest accuracy {0:.4f} (error {1:.2f}%)".format(acc, 100. * (1. - acc))


def knn_without_nn():
	"""
	Output
	------
	Running 'knn_without_nn'
	Loading data ...
	Training PCA ... Elapsed time: 10.642 sec
	Applying augmentation ... Elapsed time: 252.943 sec
	Transforming the data ... Elapsed time: 126.107 sec
	Building k-d tree ... Elapsed time: 8.022 sec
	Evaluating k-NN ... Elapsed time: 127.265 sec

	Test accuracy 0.9794 (error 2.06%)
	"""
	print "Running 'knn_without_nn'"
	print "Loading data ..."
	X_train, y_train = load_mnist(mode='train', path='data/')
	X_test, y_test = load_mnist(mode='test', path='data/')
	X_train /= 255.
	X_test /= 255.

	print_inline("Training PCA ... ")
	with Stopwatch(verbose=True):
		pca = PCA(n_components=35, whiten=True).fit(X_train)

	aug = RandomAugmentator(transform_shape=(28, 28), random_seed=1337)
	aug.add('RandomRotate', angle=(-7., 10.))
	aug.add('RandomGaussian', sigma=(0., 0.5))
	aug.add('RandomShift', x_shift=(-1, 1), y_shift=(-1, 1))
	aug.add('Dropout', p=(0., 0.2))

	print_inline("Applying augmentation ... ")
	with Stopwatch(verbose=True):
		X_train = aug.transform(X_train, 8)
		y_train = np.repeat(y_train, 9)

	print_inline("Transforming the data ... ")
	with Stopwatch(verbose=True):
		X_train = pca.transform(X_train)
		X_test = pca.transform(X_test)
		z = pca.explained_variance_ratio_[:35]
		z /= sum(z)
		alpha = 11.6
		X_train *= np.exp(alpha * z)
		X_test  *= np.exp(alpha * z)

	knn = KNNClassifier(algorithm='kd_tree', k=3, p=2, weights='uniform')
	print_inline("Building k-d tree ... ")
	with Stopwatch(verbose=True): 
		knn.fit(X_train, y_train)

	print_inline("Evaluating k-NN ... ")
	with Stopwatch(verbose=True): 
		y_pred = knn.predict(X_test)
		acc = accuracy_score(y_test, y_pred)

	print "\nTest accuracy {0:.4f} (error {1:.2f}%)".format(acc, 100. * (1. - acc))


def nn(load_nn=True):
	"""
	Output (if `load_nn` is True)
	-----------------------------
	Running 'nn'
	Loading data ...
	Loading NN ...
	Evaluating NN ... Elapsed time: 6.584 sec

	Test accuracy 0.9896 (error 1.04%)
	"""
	print "Running 'nn'"
	print "Loading data ..."
	X_train, y_train = load_mnist(mode='train', path='data/')
	X_test, y_test = load_mnist(mode='test', path='data/')
	X_train /= 255.
	X_test /= 255.
	y_test = one_hot(y_test)

	if load_nn:
		print "Loading NN ..."
		nn = load_model('models/nn.json')
	else:
		nn = _train_nn(X_train.copy(), y_train.copy())

	print_inline("Evaluating NN ... ")
	with Stopwatch(verbose=True):
		y_pred = nn.predict(X_test)
		acc = accuracy_score(y_test, y_pred)
		
	print "\nTest accuracy {0:.4f} (error {1:.2f}%)".format(acc, 100. * (1. - acc))


def logreg(load_logreg=True, load_nn=True):
	"""
	Output (if `load_logreg` is False and `load_nn` is True)
	--------------------------------------------------------
	Running 'logreg'
	Loading data ...
	Loading NN ...
	Extracting feature vectors ... Elapsed time: 30.346 sec
	Training LogReg ... Total number of parameters:  1290
	Train on 49999 samples, validate on 10001 samples

	Epoch   1/750  early stopping after 50 ................................
	elapsed:  0.70 sec - loss: 0.009 - acc.: 0.9989 - val. loss: 0.008 - val. acc.: 0.9988
	Epoch   2/750  early stopping after 50 ................................
	elapsed:  2.13 sec - loss: 0.008 - acc.: 0.9991 - val. loss: 0.009 - val. acc.: 0.9986
	Epoch   3/750  early stopping after 50 ................................
	elapsed:  3.67 sec - loss: 0.008 - acc.: 0.9991 - val. loss: 0.009 - val. acc.: 0.9986

	(...)

	elapsed: 129.7 sec - loss: 0.009 - acc.: 0.9992 - val. loss: 0.014 - val. acc.: 0.9979
	Epoch  90/750  early stopping after 2 ................................
	elapsed: 131.1 sec - loss: 0.009 - acc.: 0.9992 - val. loss: 0.015 - val. acc.: 0.9977
	Epoch  91/750  early stopping after 1 ................................
	elapsed: 132.6 sec - loss: 0.009 - acc.: 0.9994 - val. loss: 0.013 - val. acc.: 0.9985
	Elapsed time: 133.410 sec
	Evaluating LogReg ... Elapsed time: 0.173 sec

	Test accuracy 0.9899 (error 1.01%)
	"""
	print "Running 'logreg'"
	print "Loading data ..."
	X_train, y_train = load_mnist(mode='train', path='data/')
	X_test, y_test = load_mnist(mode='test', path='data/')
	X_train /= 255.
	X_test /= 255.
	y_test = one_hot(y_test)

	if load_nn:
		print "Loading NN ..."
		nn = load_model('models/nn.json')
	else:
		nn = _train_nn(X_train.copy(), y_train.copy())

	if load_logreg:
		print "Loading LogReg ..."
		logreg = load_model('models/logreg.json')
		print_inline("Extracting feature vectors ... ")
		with Stopwatch(verbose=True):
			nn.forward_pass(X_test)
			X_test = leaky_relu(nn.layers[13]._last_input)	
	else:
		print_inline("Extracting feature vectors ... ")
		with Stopwatch(verbose=True):
			nn.forward_pass(X_train)
			X_train = leaky_relu(nn.layers[13]._last_input)
			nn.forward_pass(X_test)
			X_test = leaky_relu(nn.layers[13]._last_input)	

		tts = TrainTestSplitter(shuffle=True, random_seed=1337)
		train, val = tts.split(y_train, train_ratio=50005./60000., stratify=True)
		X_val = X_train[val]
		X_train = X_train[train]
		y_val = y_train[val]
		y_train = y_train[train]
		y_train = one_hot(y_train)
		y_val = one_hot(y_val)

		logreg = LogisticRegression(n_batches=32,
									L2=0.000316,
									random_seed=1337,
									optimizer_params=dict(
										learning_rate=0.001,
										max_epochs=750,
										plot=False,
										early_stopping=50,
									verbose=True))
		print_inline("Training LogReg ... ")
		with Stopwatch(verbose=True):		
			logreg.fit(X_train, y_train, X_val=X_val, y_val=y_val)

	print_inline("Evaluating LogReg ... ")
	with Stopwatch(verbose=True):
		y_pred = logreg.predict(X_test)
		acc = accuracy_score(y_test, y_pred)
		
	print "\nTest accuracy {0:.4f} (error {1:.2f}%)".format(acc, 100. * (1. - acc))



if __name__ == '__main__':
	# Uncomment what to run:
	# ----------------------
	# knn(load_nn=True)
	# knn_without_nn()
	# nn(load_nn=True)
	# logreg(load_logreg=True, load_nn=True)
	pass