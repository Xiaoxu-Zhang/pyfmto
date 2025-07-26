import numpy as np

from ..problem import MultiTaskProblem as Mtp
from ..benchmarks import (
    Ackley,
    Griewank,
    Rastrigin,
    Rosenbrock,
    Schwefel,
    Sphere,
    Weierstrass)

class Tetci2019(Mtp):
    is_realworld = False
    intro = ""
    notes = """
        The number of tasks is either 10 or 8, depending on the input dimension. 
        If dim <= 25, 10 tasks. 
        If dim > 25, 8 tasks (due to Weierstrass function only support 1 <= dim <= 25) 
        All tasks share the same dimensionality. 
        Tasks differ by shift vectors applied to their base functions. 
    """
    references = [
        """
        CHEN Y, ZHONG J, FENG L, et al. An Adaptive Archive-Based Evolutionary
        Framework for Many-Task Optimization[J/OL]. IEEE TETCI, 2020: 369-384.
        DOI:10.1109/tetci.2019.2916051.
        """
    ]
    def __init__(self, dim=10, **kwargs):
        if not 1 <= dim <= 50:
            raise ValueError('dim must be in [1, 50]')
        super().__init__(dim, **kwargs)

    def get_info(self):
        category = ['Easy']*4 + ['Complex']*6
        assisted_task = [None]*4 + ['T1', 'T2', 'T3,T4', None, 'T4', None]
        if self.task_num == 8:
            del category[6], category[3]
            del assisted_task[6], assisted_task[3]
        return {
            "TaskID": [t.id for t in self._problem],
            "TaskName": [t.name for t in self._problem],
            "DecDim": [t.dim for t in self._problem],
            "Lower": [t.x_lb[0] for t in self._problem],
            "Upper": [t.x_ub[0] for t in self._problem],
            "Category": category,
            "Assisted By": assisted_task
        }

    def _init_tasks(self, dim, **kwargs):
        f1 = Sphere(dim, x_lb=-100, x_ub=100, **kwargs)
        f2 = Sphere(dim, x_lb=-100, x_ub=100, **kwargs)
        f3 = Sphere(dim, x_lb=-100, x_ub=100, **kwargs)
        f4 = Weierstrass(dim, x_lb=-0.5, x_ub=0.5, **kwargs)
        f5 = Rosenbrock(dim, x_lb=-50, x_ub=50, **kwargs)    # Ideal Assisted Task f1
        f6 = Ackley(dim, x_lb=-50, x_ub=50, **kwargs)        # Ideal Assisted Task f2
        f7 = Weierstrass(dim, x_lb=-0.4, x_ub=0.4, **kwargs) # Ideal Assisted Task f3,f4
        f8 = Schwefel(dim, x_lb=-500, x_ub=500, **kwargs)
        f9 = Griewank(dim, x_lb=-100, x_ub=100, **kwargs)    # Ideal Assisted Task f4
        f10 = Rastrigin(dim, x_lb=-50, x_ub=50, **kwargs)

        functions = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
        [functions[idx].set_id(idx+1) for idx in range(len(functions))]

        s9 = np.ones(dim)
        s9[:dim // 2] = -80
        s9[s9==1] = 80

        s10 = np.ones(dim)
        s10[: dim//2] = 40
        s10[s10==1] = -40

        # Here, we set the f8 shift value to 0; if we set it
        # to 420.9687, which is the setting from the reference
        # paper and also represents the global optimality of
        # f8, the global optimum will fall outside the search
        # space (-500, 500).
        shifts = [0, 80, -80, -0.4, 0, 40, -0.4, 0, s9, s10]

        if dim > 25:
            del functions[6], functions[3]
            del shifts[6], shifts[3]

        [func.set_transform(shift=shift) for func, shift in zip(functions, shifts)]

        return functions