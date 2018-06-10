test:
	nosetests --config .noserc
	rm -rf 'ml_mnist/knn.json'
	rm -rf 'ml_mnist/knn/knn.json'
	rm -rf 'ml_mnist/decomposition/pca.json'
	rm -rf 'ml_mnist/gp/gp.json'
	rm -rf 'ml_mnist/nn/rbm/rbm.json'

clean:
	find . -name '*.pyc' -type f -delete
	find . -name 'epoch*.png' -type f -delete
	find ml_mnist/nn/ -name 'epoch*.png' -type f -delete
	rm -rf cover/
	rm -rf .coverage
	rm -rf cv_models/*
