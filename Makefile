refresh: clean build install

build:
	python -m build	

install: 
	pip install -e .

build_dist:
	make clean
	python -m build
	pip install dist/*.whl
	make test

release:
	python -m twine upload dist/*

lint:
	flake8 src/ tests/ --count --max-line-length=127 --ignore=W503,F403,F405,F401,E704
	mypy src/ --follow-imports=skip --ignore-missing-imports

test:
	python -m unittest

clear-cache:
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf src/pyfmto/algorithms/__pycache__
	rm -rf src/pyfmto/experiments/__pycache__
	rm -rf src/pyfmto/framework/__pycache__
	rm -rf src/pyfmto/problems/__pycache__
	rm -rf src/pyfmto/utilities/__pycache__
	rm -rf tests/experiments/__pycache__
	rm -rf tests/framework/__pycache__
	rm -rf tests/problems/__pycache__
	rm -rf tests/utilities/__pycache__

clean:
	make clear-cache
	rm -rf build
	rm -rf dist
	rm -rf out
	rm -rf src/pyfmto.egg-info
	pip uninstall -y pyfmto || true
