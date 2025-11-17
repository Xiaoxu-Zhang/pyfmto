```

                               ____                __         
            ____     __  __   / __/  ____ ___     / /_   ____ 
           / __ \   / / / /  / /_   / __ `__ \   / __/  / __ \
          / /_/ /  / /_/ /  / __/  / / / / / /  / /_   / /_/ /
         / .___/   \__, /  /_/    /_/ /_/ /_/   \__/   \____/ 
        /_/       /____/                                      


```

# PyFMTO

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

Python 3.9+

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
   ├── run [-c/--config <config_file>]
   ├── report [-c/--config <config_file>]
   ├── list algorithms/problems/reports
   └── show <result of list>
```

Specifically, the detail and examples of the CLI commands are as follows:

- `pyfmto run`
  - Run experiments
  - You can specify a different config file by using the `-c` or `--config` option. For example, to 
    run the experiments using the config file `my_conf.yaml`, you can use the following command:
  ```bash
  pyfmto run -c my_conf.yaml
  ```
- `pyfmto report`
  - Analyze results
  - Same as `run` command, it also supports `-c/--config` option.
- `pyfmto list`
  - List available options
  - You can use the following command to list all available options for `algorithms`, `problems` or 
    `report formats`, for example:
  ```bash
  pyfmto list algorithms
  ```
- `pyfmto show`
  - Show default config of an algorithm, problem or report format
  - `show` supports all the options list by `list` command, for example:
  ```bash
  ~$ pyfmto list algorithms
  ALG1
  ALG2
  ~$ pyfmto show ALG1
  client:
    alpha: 0.1
  server:
    beta: 0.2
  ```
  - You can check the supported args of an algorithm, problem or report format through the
    `show` command.
- get help (you can also use `-h` to replace `--help` in the following commands)
  - `pyfmto --help`
  - `pyfmto <command> --help`

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

## About fmto

The repository [fmto](https://github.com/Xiaoxu-Zhang/fmto) is the official collection of 
published FMTO algorithms and serves as a practical example of how to structure and perform 
experiments. The repository includes the following components:

- A collection of published FMTO algorithms.
- A config file (config.yaml) that provides guidance on how to set up and configure the experiments.
- A template algorithm named "ALG" that you can use as a basis for implementing your own algorithm.
- A template problem named "PROB" that you can use as a basis for implementing your own problem.

The `config.yaml`, `ALG` and `PROB` provided detailed instructions, you can even start your 
research without additional documentation.
The fmto repository is currently in the early stages of development. We are actively working 
on improving existing algorithms and adding new algorithms.

## Structure of PyFMTO

<div align="center">
  <img src="https://github.com/Xiaoxu-Zhang/zxx-assets/raw/main/pyfmto-structure.svg" width="50%">
</div>

Where the filled area represents the fully developed modules. And the non-filled area represents
the base modules that can be inherited and extended.

The bottom layer listed the core technologies used in PyFMTO for computing, communicating, plotting 
and testing.

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
        # We duplicate MySTP for 10 here as an example
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

see [contributing](CONTRIBUTING.md) for instructions on how to contribute to PyFMTO.
