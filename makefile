test:
	nosetests
	rm -rf 'knn.json'
	rm -rf 'knn/knn.json'

clean:
	find . -name '*.pyc' -type f -delete
	rm -rf cover/
	rm -rf .coverage
	rm -rf cv_models/*