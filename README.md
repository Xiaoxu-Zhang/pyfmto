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

## Install

Require python 3.9+

```bash
pip install https://pyfmto.oss-cn-hangzhou.aliyuncs.com/dist/pyfmto-0.0.1-py3-none-any.whl
```

## Usage

To begin with, we highly recommend that you clone the 
[fmto](https://github.com/Xiaoxu-Zhang/fmto) repository. This repository is the official 
collection of published FMTO algorithms and serves as a practical example of how to structure 
and perform experiments. The repository includes the following components:

- A collection of published FMTO algorithms.
- A config file (config.yaml) that provides guidance on how to set up and configure the experiments.
- A template algorithm named "ALG" that you can use as a basis for implementing your own algorithm.
- A template problem named "PROB" that you can use as a basis for implementing your own problem.

> **Note**: 
> 1. The `config.yaml`, `ALG` and `PROB` provided detailed instructions, you can even start your 
> research without additional documentation.
> 2. The fmto repository is currently in the early stages of development. We are actively working 
> on improving existing algorithms and adding new algorithms.

To clone the fmto, you can use the following command:

```bash
git clone https://github.com/Xiaoxu-Zhang/fmto
```

Now, have a try! Start the experiments by the following command:

```bash
cd fmto
pyfmto run
```

Finally, analyze the results by running the following command:

```bash
pyfmto report
```

> **Note**: You can specify a different config file by using the `-c` option. For example, to run 
> the experiments using the config file `my_conf.yaml`, you can use the following command:
> 
> ```bash
> pyfmto run -c my_conf.yaml
> ```
> The report command also supports the `-c` option.

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
    
    def _init_tasks(self, dim, **kwargs) -> Union[list[SingleTaskProblem], tuple[SingleTaskProblem]]:
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

The following parameters are available for all problems and can be optionally customized:

- `fe_init`: int $\in [1, +\infty]$ (default: `5*dim`)
- `fe_max`: int $\in [\text{fe_init}, +\infty)$ (default: `11*dim`)
- `np_per_dim`: int $\in [1, +\infty)$ (default: `1`)
- `random_ctrl`: str $\in$ {'no', 'weak', 'strong'}

Available problems and their configurable parameters are listed below:

- **Synthetic**
  - **arxiv2017**
    - `dim`: int $\in [1, 50]$  # If dim > 25, the number of tasks will be 17, else 18
  - **tevc2024**
    - `dim`: int $\in [1, 10]$
    - `src_problem`: str $\in$ ['Griewank', 'Rastrigin', 'Ackley', 'Schwefel', 'Sphere', 
      'Rosenbrock', 'Weierstrass', 'Ellipsoid']
  - **tetci2019**
    - `dim`: int $\in [1, 50]$ # If dim > 25, the number of tasks will be 8, else 10
  - **cec2022**
    - `dim`: int $\in$ {10, 20}

- **Realworld**
  - **svm_landmine**

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

Coming soon...

## Contributing

see [contributing](CONTRIBUTING.md) for instructions on how to contribute to PyFMTO.
