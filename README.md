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

Require python 3.9+

```bash
pip install https://pyfmto.oss-cn-hangzhou.aliyuncs.com/dist/pyfmto-0.0.1-py3-none-any.whl
```

## Usage
For a quick start, you may want to clone the [fmto](https://github.com/Xiaoxu-Zhang/fmto) project, which 
contains several published algorithms and a template that you can implement your algorithm based on.

```bash
git clone https://github.com/Xiaoxu-Zhang/fmto
```

The structure of a pyfmto-based project should be like this:

```text
path/to/your/project/
   ├── algorithms/
   │   ├── ALG1/
   │   └── ALG2/
   │       ├── __init__.py
   │       ├── alg2_client.py
   │       └── alg2_server.py
   ├── problems/
   ├── config.yaml
   ├── run.py
   └── report.py
```

After cloning/create the project, you can start your experiments by running the following command:

```bash
cd path/to/your/project
python run.py
```

And analyze the results by running the following command:

```bash
python report.py
```

### Implement an algorithm

To implement an algorithm, you should create a directory, such as `algorithms/ALG`. Basically, the directory should 
contain the following modules:

- `__init__.py`: This file should import the implemented Client and Server classes.
- `alg_client.py`: This file should implement the client-side of the algorithm.
- `alg_server.py`: This file should implement the server-side of the algorithm.

The following is an example of an algorithm implementation. Check out the template in the `fmto` project for more details.

```python
# alg_client.py
from pyfmto import Client

class MyClient(Client):
	"""
	gamma: 0.4
	omega: 1.3
	"""
    # The parameters defined in the docting can be configured in the config file.
	def __init__(self, problem, **kwargs):
		super().__init__(problem)
		kwargs = self.update_kwargs(kwargs)
		self.gamma = kwargs['gamma']
		self.omega = kwargs['omega']
	
	def optimize():
		# implement the optimizer
		pass
```

```python
# alg_server.py
from pyfmto import Server

class MyServer(Server):
	"""
	alpha: 0.1
	beta: 0.3
	"""
	def __init__(self, **kwargs):
		super().__init__():
		kwargs = self.update_kwargs(kwargs)
		self.alpha = kwargs['alpha']
		self.beta = kwargs['beta']
	
	def aggregate(self) -> None:
		# implement the aggregate logic
		pass

	def handle_request(self, pkg) -> Any:
		# handle the requests of clients
		pass
```

```python
# __init__.py
from .alg_client import MyClient
from .alg_server import MyServer
```

### Implement a problem

To implement a problem, you should implement the following modules in the `problems/` directory:

- `__init__.py`: This file should import the implemented Problem class.
- `prob.py`: This file should implement the problem.

A demo of a problem implementation is as follows:

- `__init__.py`:
  ```python
  # __init__.py
  from .prob import MyMTP
  ```
- `prob.py`:
  ```python
  # prob.py
  import numpy as np
  from numpy import ndarray
  from pyfmto.problems import SingleTaskProblem, MultiTaskProblem
  from typing import Union
  
  
  class MySTP(SingleTaskProblem):
  
      def __init__(self, dim=2, **kwargs):
          super().__init__(dim=dim, obj=1, lb=0, ub=1, **kwargs)
  
      def _eval_single(self, x: ndarray):
          return np.sum(x)
  
  
  class MyMTP(MultiTaskProblem):
      is_realworld = False
      intro = "user defined MTP"
      notes = "a demo of user-defined MTP"
      references = ['ref1', 'ref2']
  
      def __init__(self, dim=10, **kwargs):
          super().__init__(dim, **kwargs)
  
      def _init_tasks(self, dim, **kwargs) -> Union[list[SingleTaskProblem], tuple[SingleTaskProblem]]:
          return [MySTP(dim=dim, **kwargs) for _ in range(10)]
  ```

### Configure experiments

To configure experiments, you should create a `config.yaml` file in the root directory of your project. 
The `config.yaml` file should contain the following information:

```yaml
launcher:
  results: out/results                  # Opitonal
  repeat: 20                            # Opitonal
  save: true                            # Opitonal
  loglevel: INFO                        # Opitonal
  algorithms: [ALG1, ALG2, ...]         # Required
  problems: [prob1, prob2, ...]         # Required

reporter:
  results: out/results                  # Opitonal
  algorithms:                           # Required
    - [ALG1, ALG2, ALG3]
    - [ALG1_A, ALG1_B, ALG1_C, ALG1]
  problems: [prob1, prob2]              # Required

problems:                               # Optional
  prob1:
    dim: [10, 20]
    fe_init: 50
    fe_max: 110

algorithms:                             # Optional
  ALG1:
    client:
      alpha: 0.7  # The key of the parameter should be defined in the Class's docstring
    server:
      gamma: 1.2
      omega: 0.9
  ALG1_A:  # Rename for an algorithm's variant
    base: ALG1  # Specify the base algorithm
    client:
      alpha: 0.3
```

### Entrance scripts

If you create a project by yourself, you should create `run.py` and `report.py` in the root directory of your project.

- `run.py`:
  ```python
  # run.py
  from pyfmto.experiments import Launcher
  
  if __name__ == '__main__':
      launcher = Launcher()
      launcher.run()
  ```
- `report.py`:
  ```python
  # report.py
  from pyfmto.experiments import Reports
  
  if __name__ == '__main__':
      reports = Reports()
      reports.to_curve(on_log_scale=True)
      # reports.to_excel()
      # reports.to_violin()
      # reports.to_latex()
  ```

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
