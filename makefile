test:
	nosetests

clean:
	find . -name '*.pyc' -type f -delete
	rm -rf cover/
	rm -rf .coverage