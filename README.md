```

                               ____                __         
            ____     __  __   / __/  ____ ___     / /_   ____ 
           / __ \   / / / /  / /_   / __ `__ \   / __/  / __ \
          / /_/ /  / /_/ /  / __/  / / / / / /  / /_   / /_/ /
         / .___/   \__, /  /_/    /_/ /_/ /_/   \__/   \____/ 
        /_/       /____/                                      


```

# Federated Many-task Optimization Library for Python

**pyfmto** is a Python library for federated many-task optimization research

## Install

Require python >= 3.9

```bash
pip install https://pyfmto.oss-cn-hangzhou.aliyuncs.com/dist/pyfmto-0.0.1-py3-none-any.whl
```

## Usage

First, export a demo in your project directory

```bash
cd path/to/your/project
python # enter python interactive mode
from pyfmto import export_demo
export_demo('DEMO')
```

then, your project structure should be like this:

```text
path/to/your/project/
  ├── algorithms/
  │   └── DEMO/
  │       ├── __init__.py
  │       ├── demo_client.py
  │       └── demo_server.py
  ├── config.yaml
  ├── run.py
  └── report.py
```

start the demo in cmd(Windows) or terminal(MacOS/Linux)

- Run `run.py` in IDE (PyCharm, VSCode, etc.)
- Run in terminal or cmd

```bash
cd path/to/your/project
python run.py
```

remark:

- algorithms/DEMO is the package of the algorithm
- config.yaml is the configuration file for the experiment
- run.py is the entry point for the experiment
- report.py is the report tool to generate experiment reports

## Other usage

### Problems

List available problems

```python
from pyfmto.problems import list_problems

# list all problems in console
list_problems(print_it=True)

# it also return the list of problems
prob_lst = list_problems()
```

Load a problem

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

# show distribution of init solutions in 2d space, if dim>2, only the first two dimensions will be shown
prob.plot_distribution(f'distribution plot.png')

# visualize one of the tasks (require problem dim>=2)
task = prob[0]
task.plot_2d(f'visualize2D T{first_task.id}')
task.plot_3d(f'visualize3D T{first_task.id}')
task.iplot_3d() # interactive plotting
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
    - `src_problem`: str $\in$ ['Griewank', 'Rastrigin', 'Ackley', 'Schwefel', 'Sphere', 'Rosenbrock', 'Weierstrass', 'Ellipsoid']
  - **tetci2019**
    - `dim`: int $\in [1, 50]$ # If dim > 25, the number of tasks will be 8, else 10
  - **cec2022**
    - `dim`: int $\in$ {10, 20}

- **Realworld**
  - **svm_landmine**

### Algorithms

List available algorithms

```python
from pyfmto.algorithms import list_algorithms

list_algorithms(print_it=True)
```

If you have export or implements an algorithm (e.g. `MYALG`), you will see it in the output.

```txt
builtins:
    FDEMD
    FMTBO
yours:
    MYALG
```
