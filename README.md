```

                               ____                __         
            ____     __  __   / __/  ____ ___     / /_   ____ 
           / __ \   / / / /  / /_   / __ `__ \   / __/  / __ \
          / /_/ /  / /_/ /  / __/  / / / / / /  / /_   / /_/ /
         / .___/   \__, /  /_/    /_/ /_/ /_/   \__/   \____/ 
        /_/       /____/                                      


```

# Federated Many-task Optimization Library for Python

**pyfmto** is a Python library for federated many-task optimization

## Install

Require python >= 3.9

```bash
pip install https://pyfmto.oss-cn-hangzhou.aliyuncs.com/dist/pyfmto-0.1.0-py3-none-any.whl
```

## Usage

- [Problems](#problems): A collection of synthetic and real-world problems for federated many-task optimization.
- [Algorithms](#show-available-algorithms): A collection of federated many-task optimization algorithms.
- [Framework](#implement-an-algorithm-named-myalg): A framework for implementing federated many-task optimization algorithms.
- [Experiments](#run-algorithms): Tools for running experiments and analyzing results.

### Problems

List problems

```python
from pyfmto.problems import list_problems

# list all problems in console and get name list
prob_names = list_problems()

```

Load a problem

```python
from pyfmto.problems import load_problem, list_problems

# load the first problem in the listed problems
prob_names = list_problems()
_ = load_problem(name=prob_names[0])

# `name` is case-insensitive and ignores underscores to match 'PascalCase' or 'camelCase' class name.
# So the problem `SvmLandmine` can be loaded using any of the following:
_ = load_problem('svm_landmine')
_ = load_problem('svmlandmine')
_ = load_problem('SvmLandmine')
_ = load_problem('SVM_Landmine')

# load a problem with customized args
prob = load_problem('arxiv2017', dim=2, init_fe=20, max_fe=50, np_per_dim=5)

# show problem information
print(prob)

# show distribution of init solutions in 2d space, if dim>2, only the first two dimensions will be shown
prob.show_distribution(f'distribution plot.png')

# visualize one of the tasks (require problem dim>=2)
first_task = prob[0]
first_task.plot_2d(f'visualize2D T{first_task.id}')
first_task.plot_3d(f'visualize3D T{first_task.id}')
```

Problem supported args can be found [here](#problem-args).

### Algorithms

#### Show available algorithms

```python
from pyfmto.algorithms import list_algorithms

list_algorithms(print_it=True)
```

If you have implemented an algorithm named `MYALG` [in this way](#implement-an-algorithm-named-myalg), you will see it in the console.

```txt
builtins:
    FDEMD
    FMTBO
yours:
    MYALG
```

### Framework

#### Implement an algorithm named `MYALG`

- Organize your code as follows:

```txt
path/to/your/project/
    ├── algorithms/
    │   └── MYALG
    │       ├── __init__.py
    │       ├── myalg_client.py
    │       └── myalg_server.py
    ├── run.py
    └── settings.yaml
```

- Implement the algorithm `MYALG` (Details coming soon)

```python
# myalg_server.py
from pyfmto.framework import Server, ClientPackage, ServerPackage


class MyServer(Server):

    def __init__(self):
        super().__init__()

    def handle_request(self, client_data: ClientPackage) -> ServerPackage:
        pass

    def aggregate(self, client_id):
        pass
```

```python
# myalg_client.py
from pyfmto.framework import Client


class MyClient(Client):

    def __init__(self, problem):
        super().__init__(problem)
        ...

    def optimize(self):
        pass
```

```python
# __init__.py
from .myalg_client import MyClient
from .myalg_server import MyServer
```

- Check if your algorithm can be load [in this way](#show-available-algorithms)
- Launch experiments by following the [configuration](#run-algorithms) above

### Experiments

#### Configure

Configure the experiments

```yaml
# settings.yaml
runs: # conf for experiments
  others:
    num_runs: 1
    save_res: False
    clean_tmp: True
  algorithms: [FDEMD, FMTBO]
  problems:
    cec2022:
      args:
        np_per_dim: [1, 2]
    arxiv2017:
analyses: # conf for result analysis
  results: ~
  algorithms:
    - [FMTBO, FDEMD]
  problems: [CEC2022]
  np_per_dim: [1, 2, 4, 6]
```

Edit run.py

```python
# run.py
from pyfmto.experiments import exp

exp.run()
```

#### Start experiments

- Run `run.py` in IDE
- Run in terminal or cmd

  ```bash
  cd path/to/your/project
  python run.py
  ```

## Appendix

### Problem args

The following parameters are available for all problems and can be optionally customized:

- `fe_max`: int $\in [1, +\infty)$ (default: `11*dim`)
- `fe_init`: int $\in [1, \text{fe_max}]$ (default: `5*dim`)
- `np_per_dim`: int $\in [1, +\infty)$ (default: `1`)
- `random_ctrl`: str $\in$ {'no', 'weak', 'strong'}

Available problems and their configurable parameters are listed below:

- **Synthetic**
  - **arxiv2017**
    - `dim`: int $\in [1, 50]$  # If dim > 25, the number of tasks will be 17, else 18
  - **tevc2024**
    - `dim`: int $\in [1, 10]$
    - `src_problem`: str $\in$ ['Griewank', 'Rastrigin', 'Ackley', 'Schwefel', 'Sphere', 'Rosenbrock', 'Weierstrass', 'Ellipsoid']
  - **tetci2019**
    - `dim`: int $\in [1, 50]$ # If dim > 25, the number of tasks will be 8, else 10
  - **cec2022**
    - `dim`: int $\in$ {10, 20}

- **Realworld**
  - **svm_landmine**

### Algorithm args

...