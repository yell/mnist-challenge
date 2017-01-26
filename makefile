test:
	nosetests
	rm -rf 'ml1_mnist/knn.json'
	rm -rf 'ml1_mnist/knn/knn.json'
	rm -rf 'ml1_mnist/decomposition/pca.json'

clean:
	find . -name '*.pyc' -type f -delete
	# find . -name 'epoch*.png' -type f -delete
	rm -rf cover/
	rm -rf .coverage
	rm -rf cv_models/*
	rm -rf ml1_mnist/nn/epoch*.png