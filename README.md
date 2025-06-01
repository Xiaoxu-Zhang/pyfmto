```

                               ____                __         
            ____     __  __   / __/  ____ ___     / /_   ____ 
           / __ \   / / / /  / /_   / __ `__ \   / __/  / __ \
          / /_/ /  / /_/ /  / __/  / / / / / /  / /_   / /_/ /
         / .___/   \__, /  /_/    /_/ /_/ /_/   \__/   \____/ 
        /_/       /____/                                      


```

# Federated Many-task Optimization Library for Python

## System Requirements

- Linux/MacOS
- Python 3.9+

## Getting Started

### Install

Install with pip:

```bash
pip install https://pyfmto.oss-cn-hangzhou.aliyuncs.com/dist/pyfmto-0.1.0-py3-none-any.whl
```

Or installing from the source

```bash
git clone https://github.com/pyfmto/pyfmto.git
cd pyfmto
pip install .
```

### Usage

Show available problems and load a problem

```python
from pyfmto.problems import list_problems, load_problem

# list all problems in console and get name list
prob_names = list_problems()

# load the first problem
prob = load_problem(prob_name=prob_names[0])

# `prob_name` is case-insensitive and ignores underscores to match 'PascalCase' or 'camelCase' class name.
# For example, the problem `SvmLandmine` can be loaded using any of the following:
_ = load_problem('svm_landmine')
_ = load_problem('svmlandmine')
_ = load_problem('SvmLandmine')
_ = load_problem('SVM_Landmine')

```

Play with a problem

```python
from pyfmto.problems import load_problem

# load a problem with default args
prob = load_problem(prob_name='arxiv2017')
print(prob)


# load a problem with customized args
prob = load_problem('arxiv2017', dim=2, init_fe=20, max_fe=50, np_per_dim=2)

# visualize one of the tasks (require problem dim>=2)
first_task = prob[0]
first_task.visualize_2d(f'visualize2D T{first_task.id}')
first_task.visualize_3d(f'visualize3D T{first_task.id}')

# show distribution of init solutions in 2d space, if dim>2, only the first two dimensions will be shown
prob = load_problem('arxiv2017', dim=2)
for i in range(3):
    # prob.init_solutions('no')  # choices: no, weak, strong
    prob.show_distribution(f'distribution plot {i + 1}.png')
```

Implement a federated optimization algorithm

```python
# server.py
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
# client.py
from pyfmto.framework import Client


class MyClient(Client):

    def __init__(self, problem):
        super().__init__(problem)
        ...

    def optimize(self):
        pass
```

## Problems

The following parameters are available for all problems and can be optionally customized:

- `fe_max`: int $\in [1, +\infty)$ (default: `11*dim`)
- `fe_init`: int $\in [1, \text{fe_max}]$ (default: `5*dim`)
- `np_per_dim`: int $\in [1, +\infty)$ (default: `1`)

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
    - `dim`: int $\in \{10, 20\}$

- **Realworld**
  - **svm_landmine**