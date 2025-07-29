import numpy as np
from pathlib import Path
from scipy.io import loadmat
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from ..problem import (MultiTaskProblem as Mtp,
                       SingleTaskProblem as Stp)

__all__ = ["SvmLandmine"]


class _SingleSvmProblem(Stp):
    def __init__(self, dataset: tuple, **kwargs):
        super().__init__(dim=2, obj=1, lb=np.array([1e-4, 1e-2]), ub=np.array([10.0, 10.0]), **kwargs)
        self.x_train, self.x_test, self.y_train, self.y_test = dataset
        self.data_size = len(self.x_train)
        self.set_x_global(None)

    def evaluate(self, x):
        # training svm and get the prediction with real datasets
        _x = self.before_eval(x)
        y = np.array([self._eval_single(xi) for xi in _x])
        y = self.after_eval(x, y)
        return y

    def _eval_single(self, x):
        clf = svm.SVC(kernel="rbf", C=x[0], gamma=x[1], probability=True)
        clf.fit(self.x_train, self.y_train)
        pred = clf.predict_proba(self.x_test)
        score = roc_auc_score(self.y_test, pred[:, 1])
        return 1 - score


class SvmLandmine(Mtp):
    is_realworld = True
    intro = """
        A multitask Svm Landmine Detection hyperparameter optimization problem based on the Landmine dataset.
        Each task corresponds to training an SVM model with RBF kernel, optimizing hyperparameters
        (C and gamma) to minimize the classification error (1 - AUC score).
        """
    references = [
        """
        Qi, Y., Liu, D., Dunson, D., & Carin, L. (2008). Multitask compressive sensing
        with Dirichlet process priors. Proceedings of the 25th International Conference
        on Machine Learning - ICML'08, 768–775. https://doi.org/10.1145/1390156.1390253.
        """,
        """
        Zhu, H., Wang, X., & Jin, Y. (2023). Federated Many-Task Bayesian
        Optimization. IEEE Transactions on Evolutionary Computation, 1–1.
        https://doi.org/10.1109/TEVC.2023.3279775
        """
    ]

    def __init__(self, **kwargs):
        super().__init__(True, **kwargs)

    def _init_tasks(self, *args, **kwargs):
        dataset_path = Path(__file__).parents[1] / 'datasets' / 'svm_landmine' / "LandmineData.mat"
        landmine_data = loadmat(str(dataset_path))
        features = landmine_data["feature"][0]
        label = landmine_data["label"][0]

        tasks = []
        for i in range(len(features)):
            dataset = train_test_split(features[i], label[i], test_size=0.5, stratify=label[i], random_state=0)
            dataset = (dataset[0], dataset[1], dataset[2].squeeze(), dataset[3].squeeze())
            t = _SingleSvmProblem(dataset, **kwargs)
            t.set_id(i + 1)
            tasks.append(t)
        return tasks


    def get_info(self):
        return {
            "DecDim": [2],
            "Params": ["C, gamma"],
            "Lower": ['[1e-4, 1e-2]'],
            "Upper": ['[10, 10]']
        }
