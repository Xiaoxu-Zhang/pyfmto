reinstall: clean build install lint
unittest: reinstall utest
pytest: reinstall ptest

build:
	python -m build	

install: 
	pip install dist/*.whl

release:
	python -m twine upload dist/*

lint:
	flake8
	mypy

utest:
	coverage run -m unittest
	coverage html
	coverage report -m

ptest:
	pytest
	coverage report -m

clean:
	rm -rf coverage
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf src/pyfmto/__pycache__
	rm -rf src/pyfmto/experiments/__pycache__
	rm -rf src/pyfmto/framework/__pycache__
	rm -rf src/pyfmto/problems/__pycache__
	rm -rf src/pyfmto/utilities/__pycache__
	rm -rf tests/__pycache__
	rm -rf tests/experiments/__pycache__
	rm -rf tests/framework/__pycache__
	rm -rf tests/problems/__pycache__
	rm -rf tests/utilities/__pycache__
	rm -rf build
	rm -rf dist
	rm -rf out
	rm -rf src/pyfmto.egg-info
	pip uninstall -y pyfmto || true
