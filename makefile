test:
	nosetests
	rm -rf 'knn.json'

clean:
	find . -name '*.pyc' -type f -delete
	rm -rf cover/
	rm -rf .coverage