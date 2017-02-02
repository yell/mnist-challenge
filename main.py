# AT THE VERY BOTTOM OF THIS FILE UNCOMMENT WHAT ALGORITHM YOU WANT TO RUN

import numpy as np

from ml1_mnist.knn import KNNClassifier
from ml1_mnist.decomposition import PCA
from ml1_mnist.augmentation import RandomAugmentator
from ml1_mnist.metrics import accuracy_score
from ml1_mnist.utils import Stopwatch, print_inline
from ml1_mnist.utils.dataset import load_mnist
from ml1_mnist.utils.read_write import load_model


def knn():
	"""
	Output
	------
	Loading data ...
	Training PCA ... Elapsed time: 10.642 sec
	Applying augmentation ... Elapsed time: 252.943 sec
	Transforming the data ... Elapsed time: 126.107 sec
	Building k-d tree ... Elapsed time: 8.022 sec
	Evaluating k-NN ... Elapsed time: 127.265 sec

	Test accuracy 0.9794 (error 2.06%)
	"""
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


if __name__ == '__main__':
	knn()