# PyFMTO

```text

                       ____                __         
    ____     __  __   / __/  ____ ___     / /_   ____ 
   / __ \   / / / /  / /_   / __ `__ \   / __/  / __ \
  / /_/ /  / /_/ /  / __/  / / / / / /  / /_   / /_/ /
 / .___/   \__, /  /_/    /_/ /_/ /_/   \__/   \____/ 
/_/       /____/                                      


```

[![build](https://github.com/Xiaoxu-Zhang/pyfmto/workflows/build/badge.svg)](https://github.com/Xiaoxu-Zhang/pyfmto/actions?query=workflow%3Abuild)
[![coverage](https://img.shields.io/codecov/c/github/Xiaoxu-Zhang/pyfmto)](https://codecov.io/gh/Xiaoxu-Zhang/pyfmto)
[![license](https://img.shields.io/github/license/Xiaoxu-Zhang/pyfmto)](https://github.com/Xiaoxu-Zhang/pyfmto/blob/master/LICENSE)
[![commit](https://img.shields.io/github/last-commit/Xiaoxu-Zhang/pyfmto)](https://github.com/Xiaoxu-Zhang/pyfmto/commits/main)
[![pypi](https://img.shields.io/pypi/v/pyfmto.svg)](https://pypi.org/project/pyfmto/)
[![pypi-downloads](https://img.shields.io/pepy/dt/pyfmto?label=PyPI%20downloads&color=rgb(0%2C%2079%2C%20144))](https://pypistats.org/packages/pyfmto)
[![OS Support](https://img.shields.io/badge/OS-Linux%20%7C%20MacOS%20%7C%20Windows-green)](https://pypi.org/project/pyfmto/)
[![support-version](https://img.shields.io/pypi/pyversions/pyfmto)](https://img.shields.io/pypi/pyversions/pyfmto)

**PyFMTO** is a pure Python library for federated many-task optimization research

<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/Xiaoxu-Zhang/zxx-assets/raw/main/pyfmto-demo.gif" width="95%"/><br>
      Run experiments
    </td>
    <td align="center">
      <img src="https://github.com/Xiaoxu-Zhang/zxx-assets/raw/main/pyfmto-iplot.gif" width="95%"/><br>
      Plot tasks
    </td>
  </tr>
</table>

## Table of Contents

- [Quick Start](#quick-start)
- [Command Line Interface](#command-line-interface-cli)
- [Use PyFMTO in Python](#use-pyfmto-in-python)
- [Architecture and Ecosystem](#architecture-and-ecosystem)
- [About fmto](#about-fmto)
- [Algorithm Components](#algorithm-components)
- [Problem Components](#problem-components)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [Bugs/Requests](#bugsrequests)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Quick Start

Create an environment and install PyFMTO:

```bash
conda create -n fmto python=3.10
conda activate fmto
pip install pyfmto
```

Clone the [fmto](https://github.com/Xiaoxu-Zhang/fmto.git) repository ([why?](#about-fmto)):

```bash
git clone https://github.com/Xiaoxu-Zhang/fmto.git
cd fmto
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

## Command Line Interface (CLI)

PyFMTO provides a command-line interface (CLI) for running experiments, analyzing results and
getting help. The CLI structure is as follows:

```txt
pyfmto
   ├── -h/--help
   ├── run
   ├── report
   ├── list algorithms/problems
   └── show algorithms.<alg_name>/problems.<prob_name>
```

### CLI Commands

#### Get help

```bash
pyfmto -h
pyfmto --help
pyfmto run -h
pyfmto report -h
pyfmto list -h
pyfmto show -h
```

#### Run experiments

```bash
pyfmto run              # Run with default config.yaml
pyfmto run -c config.yaml   # Run with specified config file
```

#### Generate reports

```bash
pyfmto report           # Generate reports with default config.yaml
pyfmto report -c config.yaml  # Generate with specified config file
```

#### List available components

```bash
pyfmto list algorithms  # List all available algorithms
pyfmto list problems    # List all available problems
```

#### Show component configurations

```bash
pyfmto show algorithms.<alg_name>   # Show algorithm default params
pyfmto show problems.<prob_name>    # Show problem default params
```

> **Notes**:
>
> Every subcommand supports `-c/--config <config_file>` to specify a configuration file.
> A `config.yaml` file in the current working directory is required when `-c` is not provided.
>
> In the subcommands `list` and `show`, strings 'algorithms' and 'problems' can be
> replaced with any prefix of length ≥ 1. PyFMTO matches the prefix to the corresponding category.
> For example:
>
> `pyfmto list algorithms` is equivalent to:
>
> - `pyfmto list a`
> - `pyfmto list al`
> - `pyfmto list alg`
> - ...
>
> `pyfmto show problems.<prob_name>` is equivalent to:
>
> - `pyfmto show p.<prob_name>`
> - `pyfmto show prob.<prob_name>`
> - ...

## Use PyFMTO in Python

### Basic Usage

```python
from pyfmto import Launcher, Reporter, ConfigLoader

if __name__ == '__main__':
    conf = ConfigLoader()
    launcher = Launcher(conf.launcher)
    launcher.run()
    
    reporter = Reporter(conf.reporter)
    reporter.report()
```

### Loading Problems

```python
from pyfmto import load_problem

# Initialize a problem with customized args
prob = load_problem('arxiv2017', dim=2, fe_init=20, fe_max=50, npd=5)

# Problem instance can be printed
print(prob)
```

### Reporter Output Formats

The Reporter supports multiple output formats:

```python
from pyfmto import Reporter, ConfigLoader

conf = ConfigLoader()
reporter = Reporter(conf.reporter)

# Generate performance curves
reporter.to_curve()

# Generate Excel report with statistics
reporter.to_excel()

# Generate LaTeX table
reporter.to_latex()

# Print statistics to console
reporter.to_console()

# Generate violin plots
reporter.to_violin()
```

## Architecture and Ecosystem

<div align="center">
  <img src="https://github.com/Xiaoxu-Zhang/zxx-assets/raw/main/pyfmto-architecture.svg" width="90%">
</div>

Where the filled area represents the fully developed modules, and the non-filled area represents
the base modules that can be inherited and extended.

The bottom layer lists the core technologies used in PyFMTO for computing, communicating, plotting
and testing.

## About fmto

The repository [fmto](https://github.com/Xiaoxu-Zhang/fmto) is the official collection of
published FMTO algorithms. The relationship between `fmto` and `PyFMTO` is as follows:

<p align="center">
    <img src="https://github.com/Xiaoxu-Zhang/zxx-assets/raw/main/fmto-relation.svg"/>
<p>

The `fmto` is designed to provide a platform for researchers to compare and evaluate the
performance of different FMTO algorithms. The repository is built on top of the PyFMTO library,
which provides a flexible and extensible framework for implementing FMTO algorithms.

It also serves as a practical example of how to structure and perform experiments. The repository
includes the following components:

- A collection of published FMTO algorithms
- A config file (config.yaml) that provides guidance on how to set up and configure the experiments
- A template algorithm named "DEMO" that you can use as a basis for implementing your own algorithm
- A template problem named "demo" that you can use as a basis for implementing your own problem

The `config.yaml`, `algorithms/DEMO` and `problems/demo` provide detailed instructions. You can
even start your research without additional documentation. The fmto repository is currently in
the early stages of development. I'm actively working on improving existing algorithms and adding
new algorithms.

## Algorithm Components

An algorithm includes two parts: the **Client** and the **Server**. The client is responsible for
optimizing the local problem, and the server is responsible for aggregating knowledge from
the clients.

### Client Implementation

```python
from pyfmto import Client
from pyfmto.framework import record_runtime

class MyClient(Client):
    def __init__(self, problem, **kwargs):
        super().__init__(problem)
        # Your initialization code

    def optimize(self):
        """
        Implement the optimization logic.
        This method is called iteratively until fe_available reaches 0.
        """
        # Get data from server if needed
        data = self.request_server(pkg)
        
        # Perform local optimization
        # ...
        
        # Send data to server if needed
        self.request_server(pkg)
```

### Server Implementation

```python
from pyfmto import Server

class MyServer(Server):
    def __init__(self, **kwargs):
        super().__init__()
        # Your initialization code
 
    def aggregate(self) -> None:
        """
        Implement the aggregation logic.
        This method is called periodically to aggregate client knowledge.
        """
        pass

    def handle_request(self, pkg) -> Any:
        """
        Handle requests from clients to exchange data.
        """
        pass
```

### Algorithm Registration

To register your algorithm with PyFMTO, create an AlgorithmData class:

```python
from pyfmto.framework import AlgorithmData

class MyAlgorithm(AlgorithmData):
    client = MyClient
    server = MyServer
```

## Problem Components

There are two types of problems: **SingleTaskProblem** and **MultiTaskProblem**. A SingleTaskProblem
has only one objective function. A MultiTaskProblem aggregates multiple SingleTaskProblems.

### SingleTaskProblem

```python
import numpy as np
from numpy import ndarray
from pyfmto.problem import SingleTaskProblem


class MySTP(SingleTaskProblem):

    def __init__(self, dim=2, **kwargs):
        super().__init__(dim=dim, obj=1, lb=0, ub=1, **kwargs)

    def _eval_single(self, x: ndarray):
        """
        Evaluate a single solution.
        
        Parameters
        ----------
        x : ndarray
            Input vector of shape (dim,)
            
        Returns
        -------
        float or ndarray
            Objective value(s)
        """
        return np.sum(x ** 2)
```

### MultiTaskProblem

```python
from pyfmto.problem import SingleTaskProblem, MultiTaskProblem


class MyMTP(MultiTaskProblem):
    is_realworld = False
    intro = "User defined MTP"
    notes = "A demo of user-defined MTP"
    references = ['ref1', 'ref2']

    def __init__(self, dim=10, **kwargs):
        super().__init__(dim, **kwargs)

    def _init_tasks(self, dim, **kwargs) -> list[SingleTaskProblem]:
        # Create 10 instances of MySTP as an example
        return [MySTP(dim=dim, **kwargs) for _ in range(10)]
```

> **Note**: There are some classical SingleTaskProblems defined in `pyfmto.problems.benchmarks`
> module that you can use directly.

## Visualization

### SingleTaskProblem Visualization

```python
from pyfmto.problem.benchmarks import Ackley

task = Ackley()
task.plot_2d('visualize2D.png')    # 2D contour plot
task.plot_3d('visualize3D.png')    # 3D surface plot
task.iplot_3d()                     # Interactive 3D plot
```

### MultiTaskProblem Visualization

The right side interactive plotting at the beginning of this README is generated by the following code:

```python
from pyfmto import load_problem

if __name__ == '__main__':
    prob = load_problem('arxiv2017', dim=2)
    prob.iplot_tasks_3d(tasks_id=[2, 5, 12, 18])
```

### Additional Visualization Methods

```python
from pyfmto import load_problem

prob = load_problem('arxiv2017', dim=2)

# Plot distribution of initial solutions
prob.plot_distribution(filename='distribution.png')

# Plot similarity heatmap between tasks
prob.plot_similarity_heatmap(filename='similarity.png')
```

## Contributing

See [contributing](https://github.com/Xiaoxu-Zhang/pyfmto/blob/main/CONTRIBUTING.md) for instructions on how to contribute to PyFMTO.

## Bugs/Requests

Please send bug reports and feature requests through
[github issue tracker](https://github.com/Xiaoxu-Zhang/pyfmto/issues). PyFMTO is
currently under development, and it's open to any constructive suggestions.

## License

Copyright (c) 2025 Xiaoxu Zhang

Distributed under the terms of the
[Apache 2.0 license](https://github.com/Xiaoxu-Zhang/pyfmto/blob/main/LICENSE).

## Acknowledgements

### Foundations

This project is supported, in part, by the National Natural Science Foundation of China under
Grant 62006143; the Natural Science Foundation of Shandong Province under Grants ZR2025MS1012
and ZR2020MF152. I would like to express our sincere gratitude to **Smart Healthcare and Big Data
Laboratory, Shandong Women's University**, for providing research facilities and technical support.

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
`jinja2`, `msgpack`, `openpyxl`, `opfunu`, `pillow`, `pydantic`, `pydantic_core`, `pyDOE3`,
`pyyaml`, `requests`, `ruamel-yaml`, `scienceplots`, `setproctitle`, `tabulate`, `tqdm`,
`uvicorn`, and `wrapt`.  

Your dedication to building and maintaining these tools has made it possible for this project to
achieve both depth and breadth that would otherwise have been unattainable.
