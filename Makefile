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
	flake8 src/ tests/ --exclude=src/pyfmto/algorithms/ --count --max-line-length=127
	mypy src/ --exclude=src/pyfmto/algorithms/ --follow-imports=skip --ignore-missing-imports

utest:
	coverage run -m unittest
	coverage report -m
	coverage html

ptest:
	pytest

clean:
	rm -rf coverage
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf src/pyfmto/__pycache__
	rm -rf src/pyfmto/algorithms/__pycache__
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
