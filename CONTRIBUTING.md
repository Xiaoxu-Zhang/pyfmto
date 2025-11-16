# Contributing to PyFMTO

First of all, many thanks to everyone who wants to contribute to PyFMTO! 

We welcome contributions from the community to enhance the functionality and usability of
**pyfmto**. If you encounter any issues or have suggestions for improvements, please feel free to 
open an issue or submit a pull request.

# Coding

PyFMTO follows PEP 8 style and requires type annotation for functions. Please do a lint check 
described below before you submit the code.

There's no exact coding standards PyFMTO follows, just try to make the code readable.

## Tests

If you are implementing something new, you need to write tests for it. PyFMTO is a 100% coverage 
project. There might be certain lines that can't be tested. If that's the case, just include a 
clear reason and add `pragma: no cover` after the line to skip it.

Every new line of your code should be covered by the existing tests or your own tests.

## Docs

If you implement a new feature, you also need to write docs for it for others to understand. It 
should live in [README](README.md). If you don't know where the docs belong to, ask me in the 
issue/PR.

## Build and Test

To contribute, first fork this project under your account, then create a feature branch:

```bash
git clone https://github.com/<your_user_name>/pyfmto.git
cd pyfmto
git checkout -b <you_feature_branch>
```

`conda` (or other package manager) is highly recommended for development.

```bash
conda create -n <your_env_name> python=3.10
conda activate <your_env_name>
```

Install the development dependencies:

```bash
pip install -e ".[dev]"
```

To build the project on Linux/macOS, you can simply do:

```bash
make build
```

However, if you are on Windows or prefer more explicit build process, you can do the following:

```
# uninstall pyfmto first
pip uninstall pyfmto -y
# build the project
python -m build
# install
pip install dist/*.whl
```

## Lint

Check lint with flake8 and mypy:

```bash
# On Unix-like
make lint

# Explicit or Windows
flake8
mypy
```

## Test

```bash
# On Unix-like
make pytest # or make unittest

# Explicit or Windows
pytest
coverage report -m
```

## Pull Request

Do a pull request to the `main` branch of `Xiaoxu-Zhang/pyfmto`, and I will review the code and 
give feedbacks as soon as possible.