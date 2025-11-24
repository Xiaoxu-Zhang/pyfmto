```

                               ____                __         
            ____     __  __   / __/  ____ ___     / /_   ____ 
           / __ \   / / / /  / /_   / __ `__ \   / __/  / __ \
          / /_/ /  / /_/ /  / __/  / / / / / /  / /_   / /_/ /
         / .___/   \__, /  /_/    /_/ /_/ /_/   \__/   \____/ 
        /_/       /____/                                      


```

# PyFMTO

[![build](https://github.com/Xiaoxu-Zhang/pyfmto/workflows/build/badge.svg)](https://github.com/Xiaoxu-Zhang/pyfmto/actions?query=workflow%3Abuild)
[![coverage](https://img.shields.io/codecov/c/github/Xiaoxu-Zhang/pyfmto)](https://codecov.io/gh/Xiaoxu-Zhang/pyfmto)
[![pypi](https://img.shields.io/pypi/v/pyfmto.svg)](https://pypi.org/project/pyfmto/)
[![support-version](https://img.shields.io/pypi/pyversions/pyfmto)](https://img.shields.io/pypi/pyversions/pyfmto)
[![license](https://img.shields.io/github/license/Xiaoxu-Zhang/pyfmto)](https://github.com/Xiaoxu-Zhang/pyfmto/blob/master/LICENSE)
[![commit](https://img.shields.io/github/last-commit/Xiaoxu-Zhang/pyfmto)](https://github.com/Xiaoxu-Zhang/pyfmto/commits/main)

**PyFMTO** is a pure Python library for federated many-task optimization research

<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/Xiaoxu-Zhang/zxx-assets/raw/main/pyfmto-demo.gif" 
width="95%"/><br>
      Run experiments
    </td>
    <td align="center">
      <img src="https://github.com/Xiaoxu-Zhang/zxx-assets/raw/main/pyfmto-iplot.gif" 
width="95%"/><br>
      Plot tasks
    </td>
  </tr>
</table>

## Requirements

- Linux or macOS
- Python 3.10+

> **Note**
>
> PyFMTO is developed and tested on Linux and macOS. It may not work well on Windows.

## Usage

### Quick Start

Clone the [fmto](https://github.com/Xiaoxu-Zhang/fmto.git) repository ([why?](#about-fmto)):

```bash
git clone https://github.com/Xiaoxu-Zhang/fmto.git
cd fmto
```

Create an environment (`conda` is recommended) and install PyFMTO:

```bash
conda create -n fmto python=3.10
conda activate fmto
pip install pyfmto
```

Start the experiments:

```bash
pyfmto run
```

Generate reports:

```bash
pyfmto report
```

The reports will be saved in the folder `out/results/<today>`

### Command-line Interface (CLI)

PyFMTO provides a command-line interface (CLI) for running experiments, analyzing results and 
get helps. The CLI layers are as follows:

```txt
pyfmto
   ├── -h/--help
   ├── run [-c/--config <config_file>]
   ├── report [-c/--config <config_file>]
   ├── list algorithms/problems/reports
   └── show <result of list>
```

**Examples:**

- Get help:
    ```bash
    pyfmto -h # or ↓
    # pyfmto --help
    # pyfmto list -h
    ```
- Run experiments:
    ```bash
    pyfmto run # or ↓
    # pyfmto run -c config.yaml
    ```
- Generate reports:
    ```bash
    pyfmto report # or ↓
    # pyfmto report -c config.yaml
    ```
- List something:
    ```bash
    pyfmto list algorithms
    ```
    output:
    ```txt
    Found 6 Available Algorithms:
    FDEMD
    ADDFBO
    BO
    FMTBO
    IAFFBO
    ALG
    ```
- Show supported configurations:
    ```bash
    pyfmto show ALG
    # pyfmto will automatically find the name in 'algorithms', 'problems' and 'reports'
    ```
    output:
    ```txt
    client:   
      alpha: 0.2
    
    server:
      beta: 0.5
    ```

### Use PyFMTO in python

```python
from pyfmto import Launcher, Reporter

if __name__ == '__main__':
    launcher = Launcher()
    launcher.run()
    
    reporter = Reporter()
    reporter.to_curve()
    # reporter.to_ ...
```

## Architecture and Ecosystem

<div align="center">
  <img src="https://github.com/Xiaoxu-Zhang/zxx-assets/raw/main/pyfmto-architecture.svg" 
width="90%">
</div>

Where the filled area represents the fully developed modules. And the non-filled area represents
the base modules that can be inherited and extended.

The bottom layer listed the core technologies used in PyFMTO for computing, communicating, plotting 
and testing.

## About fmto

The repository [fmto](https://github.com/Xiaoxu-Zhang/fmto) is the official collection of 
published FMTO algorithms. The relationship between the `fmto` and `PyFMTO` is as follows:

<p align="center">
    <img src="https://github.com/Xiaoxu-Zhang/zxx-assets/raw/main/fmto-relation.svg"/>
<p>

The `fmto` is designed to provide a platform for researchers to compare and evaluate the 
performance of different FMTO algorithms. The repository is built on top of the PyFMTO library, 
which provides a flexible and extensible framework for implementing FMTO algorithms.

It also serves as a practical example of how to structure and perform experiments. The repository 
includes the following components:

- A collection of published FMTO algorithms.
- A config file (config.yaml) that provides guidance on how to set up and configure the experiments.
- A template algorithm named "ALG" that you can use as a basis for implementing your own algorithm.
- A template problem named "PROB" that you can use as a basis for implementing your own problem.

The `config.yaml`, `ALG` and `PROB` provided detailed instructions, you can even start your 
research without additional documentation. The fmto repository is currently in the early stages 
of development. I'm actively working on improving existing algorithms and adding new algorithms.

## Algorithm's Components

An algorithm includes two parts: the client and the server. The client is responsible for 
optimizing the local problem and the server is responsible for aggregating the knowledge from 
the clients. The required components for client and server are as follows:

```python
# myalg_client.py
from pyfmto import Client, Server

class MyClient(Client):
	def __init__(self, problem, **kwargs):
		super().__init__(problem)

	def optimize():
		# implement the optimizer
		pass

class MyServer(Server):
	def __init__(self, **kwargs):
		super().__init__():
	
	def aggregate(self) -> None:
		# implement the aggregate logic
		pass

	def handle_request(self, pkg) -> Any:
		# handle the requests of clients to exchange data
		pass
```

## Problem's Components

There are two types of problems: single-task problems and multitask problems. A single-task 
problem is a problem that has only one objective function. A multitask problem is a problem that 
has multiple single-task problems. To define a multitask problem, you should implement several 
SingleTaskProblem and then define a MultiTaskProblem to aggregate them.

> **Note**: There are some classical SingleTaskProblem defined in `pyfmto.problems.benchmarks` 
> module. You can use them directly.

```python
import numpy as np
from numpy import ndarray
from pyfmto.problems import SingleTaskProblem, MultiTaskProblem
from typing import Union

class MySTP(SingleTaskProblem):

    def __init__(self, dim=2, **kwargs):
        super().__init__(dim=dim, obj=1, lb=0, ub=1, **kwargs)
    
    def _eval_single(self, x: ndarray):
        pass

class MyMTP(MultiTaskProblem):
    is_realworld = False
    intro = "user defined MTP"
    notes = "a demo of user-defined MTP"
    references = ['ref1', 'ref2']
    
    def __init__(self, dim=10, **kwargs):
        super().__init__(dim, **kwargs)
    
    def _init_tasks(self, dim, **kwargs) -> list[SingleTaskProblem]:
        # Duplicate MySTP for 10 here as an example
        return [MySTP(dim=dim, **kwargs) for _ in range(10)]
  ```

## Tools

### list_problems

```python
from pyfmto.problems import list_problems

# list all problems in console
list_problems(print_it=True)

# it also return the list of problems
prob_lst = list_problems()
```

### load_problem

```python
from pyfmto.problems import load_problem, list_problems

# load each problem in the list result
for prob in list_problems():
  _ = load_problem(name=prob)

# or load a problem by name, which is case-insensitive and ignores underscores.
# So the problem `arxiv2017` can be loaded using any of the following:
_ = load_problem('Arxiv2017')
_ = load_problem('arXiv2017')
_ = load_problem('ARXIV2017')
_ = load_problem('arxiv_2017')

# load a problem with customized args
prob = load_problem('arxiv2017', dim=2, fe_init=20, fe_max=50, np_per_dim=5)

# show problem information
print(prob)
```

## Visualization

### SingleTaskProblem Visualization

```python
from pyfmto.problems.benchmarks import Ackley

task = Ackley()
task.plot_2d(f'visualize2D T{first_task.id}')
task.plot_3d(f'visualize3D T{first_task.id}')
task.iplot_3d() # interactive plotting
```

### MultiTaskProblem Visualization

The right side GIF at the beginning of this README is generated by the following code:

```python
from pyfmto import load_problem

if __name__ == '__main__':
    prob = load_problem('arxiv2017', dim=2)
    prob.iplot_tasks_3d(tasks_id=[2, 5, 12, 18])
```

## Contributing

See [contributing](https://github.com/Xiaoxu-Zhang/pyfmto/blob/main/CONTRIBUTING.md) for instructions on how to contribute to PyFMTO.

## Bugs/Requests

Please send bug reports and feature requests through
[github issue tracker](https://github.com/Xiaoxu-Zhang/pyfmto/issues). PyFMTO is 
currently under development now, and it's open to any constructive suggestions.

## License

Copyright (c) 2025 Xiaoxu Zhang

Distributed under the terms of the
[Apache 2.0 license](https://github.com/Xiaoxu-Zhang/pyfmto/blob/main/LICENSE).

## Acknowledgements

### Foundations
This project is supported, in part, by the National Natural Science Foundation of China under 
Grant 62006143; the Natural Science Foundation of Shandong Province under Grants ZR2025MS1012 
and ZR2020MF152. I would like to express our sincere gratitude to Smart Healthcare and Big Data 
Laboratory, Shandong Women's University, for providing research facilities and technical support.


### Mentorship and Team Support  
I would like to express my sincere gratitude to the **Computational Intelligence and 
Applications Group** for their invaluable help, encouragement, and collaboration throughout the 
development of this project.  

Special thanks go to my mentor, [Jie Tian](https://github.com/Jetina), whose insightful guidance 
and constructive feedback were crucial in refining and improving the work at every stage.

### Open Source Contributions  
This project would not have been possible without the outstanding contributions of the 
open-source community. I am deeply grateful to the maintainers and contributors of the following 
projects:  

- **[FastAPI](https://fastapi.tiangolo.com)** – A high-performance web framework that made 
  building APIs both fast and efficient.  
- **[NumPy](https://numpy.org)** – The fundamental package for scientific computing in Python, 
  enabling high-speed numerical operations.  
- **[Pandas](https://pandas.pydata.org)** – Powerful data structures and tools that formed the 
  backbone of data analysis in this work.  
- **[Matplotlib](https://matplotlib.org)** and **[Seaborn](https://seaborn.pydata.org)** – 
  Essential for producing high-quality, publication-ready visualizations.  
- **[PyVista](https://docs.pyvista.org)** – An intuitive, high-level 3D plotting and mesh 
  analysis interface, making scientific visualization seamlessly integrated into PyFMTO.  
- **[Scikit-learn](https://scikit-learn.org)** – An extensive set of machine learning algorithms 
  and utilities.  
- **[SciPy](https://scipy.org)** – Fundamental algorithms and mathematical functions critical to 
  scientific computing.  

I would also like to acknowledge the maintainers and contributors of other open-source libraries 
that supported this work, including:  
`jinja2`, `msgpack`, `openpyxl`, `opfunu`, `pillow`, `pydantic`, `pydantic_core`, `pyDOE`, 
`pyyaml`, `requests`, `ruamel-yaml`, `scienceplots`, `setproctitle`, `tabulate`, `tqdm`, 
`uvicorn`, and `wrapt`.  

Your dedication to building and maintaining these tools has made it possible for this project to 
achieve both depth and breadth that would otherwise have been unattainable.  
