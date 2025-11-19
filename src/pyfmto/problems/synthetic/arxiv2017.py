import numpy as np
from pathlib import Path
from scipy.io import loadmat

from ..problem import MultiTaskProblem as Mtp
from ..benchmarks import (
    Ackley,
    Griewank,
    Rastrigin,
    Rosenbrock,
    Schwefel,
    Sphere,
    Weierstrass)

__all__ = ["Arxiv2017"]


class Arxiv2017(Mtp):
    """
    dim: 10  # 1 <= dim <= 50
    """
    is_realworld = False
    intro = """
        A many-task single-objective optimization problem based on transformed versions
        of classical single-objective benchmark functions. Each task corresponds to an
        individual optimization problem.
    """
    notes = """
        The number of tasks is either 17 or 18, depending on the input dimension.
        If dim <= 25, 18 tasks, If dim > 25, 17 tasks (due to Weierstrass function only support 1 <= dim <= 25)
        All tasks share the same dimensionality.
        Tasks differ by rotation matrices and shift vectors applied to their base functions.
        This benchmark was used in federated many-task Bayesian optimization experiments.
        Task10 same with task13. Task6 same with task18.
    """
    references = [
        """
        B. Da, Y.-S. Ong, L. Feng, A. K. Qin, A. Gupta, Z. Zhu, C.-K. Ting,
        K. Tang, and X. Yao, "Evolutionary multitasking for single-objective
        continuous optimization: Benchmark problems, performance metric, and
        baseline results," arXiv preprint arXiv:1706.03470, 2017.
        """,
        """
        Zhu, H., Wang, X., & Jin, Y. (2023). Federated Many-Task Bayesian
        Optimization. IEEE Transactions on Evolutionary Computation, 1–1.
        https://doi.org/10.1109/TEVC.2023.3279775
        """
    ]

    def __init__(self, dim=10, **kwargs):
        if dim is None:
            raise ValueError("dim did not specified")
        if not 1 <= dim <= 50:
            raise ValueError(f"dim must be in [1, 50], but got {dim}")
        super().__init__(dim, **kwargs)

    def get_info(self):
        return {
            "TaskID": [t.id for t in self._problem],
            "TaskName": [t.name for t in self._problem],
            "DecDim": [t.dim for t in self._problem],
            "Lower": [t.lb[0] for t in self._problem],
            "Upper": [t.ub[0] for t in self._problem]
        }

    def _init_tasks(self, dim, **kwargs):
        datasets = Path(__file__).parents[1] / 'datasets' / 'arxiv2017'
        cih = loadmat(str(datasets / 'CI_H.mat'))
        cim = loadmat(str(datasets / 'CI_M.mat'))
        cil = loadmat(str(datasets / 'CI_L.mat'))
        pih = loadmat(str(datasets / 'PI_H.mat'))
        pim = loadmat(str(datasets / 'PI_M.mat'))
        pil = loadmat(str(datasets / 'PI_L.mat'))
        nih = loadmat(str(datasets / 'NI_H.mat'))
        nim = loadmat(str(datasets / 'NI_M.mat'))
        nil = loadmat(str(datasets / 'NI_L.mat'))

        rot1, rot2 = cih['Rotation_Task1'][0:dim, 0:dim], cih['Rotation_Task2'][0:dim, 0:dim]
        rot3, rot4 = cim['Rotation_Task1'][0:dim, 0:dim], cim['Rotation_Task2'][0:dim, 0:dim]
        rot5, rot6 = cil['Rotation_Task1'][0:dim, 0:dim], None
        rot7, rot8 = pih['Rotation_Task1'][0:dim, 0:dim], None
        rot9, rot10 = pim['Rotation_Task1'][0:dim, 0:dim], None
        rot11, rot12 = pil['Rotation_Task1'][0:dim, 0:dim], pil['Rotation_Task2'][0:dim, 0:dim]
        rot13, rot14 = None, nih['Rotation_Task2'][0:dim, 0:dim]
        rot15, rot16 = nim['Rotation_Task1'][0:dim, 0:dim], nim['Rotation_Task2'][0:dim, 0:dim]
        rot17, rot18 = nil['Rotation_Task1'][0:dim, 0:dim], None

        shift1, shift2 = np.squeeze(cih['GO_Task1'])[0:dim], np.squeeze(cih['GO_Task2'])[0:dim]
        shift3, shift4 = np.squeeze(cim['GO_Task1'])[0:dim], np.squeeze(cim['GO_Task2'])[0:dim]
        shift5, shift6 = np.squeeze(cil['GO_Task1'])[0:dim], None
        shift7, shift8 = np.squeeze(pih['GO_Task1'])[0:dim], np.squeeze(pih['GO_Task2'])[0:dim]
        shift9, shift10 = np.squeeze(pim['GO_Task1'])[0:dim], None
        shift11, shift12 = np.squeeze(pil['GO_Task1'])[0:dim], np.squeeze(pil['GO_Task2'])[0:dim]
        shift13, shift14 = None, np.squeeze(nih['GO_Task2'])[0:dim]
        shift15, shift16 = np.squeeze(nim['GO_Task1'])[0:dim], np.squeeze(nim['GO_Task2'])[0:dim]
        shift17, shift18 = np.squeeze(nim['GO_Task1'])[0:dim], None

        func1 = Griewank(dim, lb=-100, ub=100, **kwargs)
        func2 = Rastrigin(dim, lb=-50, ub=50, **kwargs)
        func3 = Ackley(dim, lb=-50, ub=50, **kwargs)
        func4 = Rastrigin(dim, lb=-50, ub=50, **kwargs)
        func5 = Ackley(dim, lb=-50, ub=50, **kwargs)
        func6 = Schwefel(dim, lb=-500, ub=500, **kwargs)
        func7 = Rastrigin(dim, lb=-50, ub=50, **kwargs)
        func8 = Sphere(dim, lb=-100, ub=100, **kwargs)
        func9 = Ackley(dim, lb=-50, ub=50, **kwargs)
        func10 = Rosenbrock(dim, lb=-50, ub=50, **kwargs)
        func11 = Ackley(dim, lb=-50, ub=50, **kwargs)
        func12 = Weierstrass(dim, lb=-0.5, ub=0.5, **kwargs)
        func13 = Rosenbrock(dim, lb=-50, ub=50, **kwargs)
        func14 = Rastrigin(dim, lb=-50, ub=50, **kwargs)
        func15 = Griewank(dim, lb=-100, ub=100, **kwargs)
        func16 = Weierstrass(dim, lb=-0.5, ub=0.5, **kwargs)
        func17 = Rastrigin(dim, lb=-50, ub=50, **kwargs)
        func18 = Schwefel(dim, lb=-500, ub=500, **kwargs)

        functions = [
            func1, func2, func3, func4, func5, func6, func7, func8, func9,
            func10, func11, func12, func13, func14, func15, func16, func17, func18
        ]

        # Set id
        for f_id in range(len(functions)):
            functions[f_id].set_id(f_id + 1)

        rot_mats = [
            rot1, rot2, rot3, rot4, rot5, rot6, rot7, rot8, rot9,
            rot10, rot11, rot12, rot13, rot14, rot15, rot16, rot17, rot18
        ]

        shift_mats = [
            shift1, shift2, shift3, shift4, shift5, shift6, shift7, shift8, shift9,
            shift10, shift11, shift12, shift13, shift14, shift15, shift16, shift17, shift18
        ]

        if dim > 25:
            # Transform matrix of pil 'Rotation_Task2' and 'GO_Task2' with different shape (25,25)
            del functions[11]
            del rot_mats[11]
            del shift_mats[11]

        for f, rot_mat, shift_mat in zip(functions, rot_mats, shift_mats):
            f.set_transform(rot_mat, shift_mat)

        return functions
