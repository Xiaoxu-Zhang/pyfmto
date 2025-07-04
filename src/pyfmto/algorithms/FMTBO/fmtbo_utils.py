import numpy as np
from enum import Enum, auto
from pyDOE import lhs


def power(mat1, mat2):
    # To solve the problem: Numpy does not seem to allow fractional powers of negative numbers
    return np.sign(mat1) * np.power(np.abs(mat1), mat2)


class Actions(Enum):
    PUSH_INIT = auto()
    PULL_INIT = auto()
    PULL_UPDATE = auto()
    PUSH_UPDATE = auto()


class AggData:
    def __init__(self, version, src_num, agg_res):
        self.version = version
        self.src_num = src_num
        self.agg_res = agg_res


class GeneticAlgorithm:
    def __init__(self, x_lb, x_ub, dim, pop_size, max_gen, pb=None):
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.dim = dim
        if isinstance(x_lb, (int, float)):
            self.x_bound = (self.x_lb, self.x_ub)
        elif isinstance(x_lb, (list, tuple, np.ndarray)):
            self.x_bound = (self.x_lb[0], self.x_ub[0])
        else:
            raise ValueError(f"Type of x_lb is {type(x_lb)}, but it should be int or float or iterable")
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.pb = pb

    def optimize(self, function, max_gen=None, archive=None):
        if max_gen is not None:
            self.max_gen = max_gen
        if archive is not None:
            parents = archive[-self.pop_size:]
        else:
            parents = init_samples(lb=self.x_lb, ub=self.x_ub, dim=self.dim, n_samples=self.pop_size)
        ac_best = -10000
        x_best = None
        for _ in range(self.max_gen):
            offspring = self._generate_offspring(parents)
            popcom = np.vstack((parents, offspring))
            ac_values = function(popcom)
            ac_idx = ac_values.argsort()[::-1]
            parents = popcom[ac_idx[:self.pop_size]]
            if ac_best < ac_values[ac_idx[0]]:
                ac_best = ac_values[ac_idx[0]]
                x_best = popcom[ac_idx[0]]
        return x_best, ac_best

    def _generate_offspring(self, parents):
        pop_size, d = parents.shape
        pop_size = pop_size-1 if pop_size % 2 == 1 else pop_size
        self.pop_size = pop_size
        # crossover(simulated binary crossover)
        # dic_c is the distribution index of crossover
        dis_c = 20
        mu = np.random.rand(int(pop_size / 2), d)
        idx1 = [i for i in range(0, pop_size, 2)]
        idx2 = [i + 1 for i in range(0, pop_size, 2)]
        parent1 = parents[idx1, :]
        parent2 = parents[idx2, :]
        element_min = np.minimum(parent1, parent2)
        element_max = np.maximum(parent1, parent2)
        tmp_min = np.minimum(element_min - self.x_lb, self.x_ub - element_max)
        beta = 1 + 2 * tmp_min / np.maximum(abs(parent2 - parent1), 1e-6)
        alpha = 2 - beta ** (-dis_c - 1)
        betaq = power(alpha * mu, 1 / (dis_c + 1)) * (mu <= 1 / alpha) + \
                power(1. / (2 - alpha * mu), 1 / (dis_c + 1)) * (mu > 1. / alpha)
        # the mutation is performed randomly on each variable
        betaq = betaq * power(-1, np.random.randint(0, 2, (int(pop_size / 2), d)))
        betaq[np.random.rand(int(pop_size / 2), d) > 0.5] = 1
        offspring1 = 0.5 * ((1 + betaq) * parent1 + (1 - betaq) * parent2)
        offspring2 = 0.5 * ((1 - betaq) * parent1 + (1 + betaq) * parent2)
        pop_crossover = np.vstack((offspring1, offspring2))

        # mutation (polynomial mutation)
        # dis_m is the distribution index of polynomial mutation
        dis_m = 20
        pro_m = 1
        rand_var = np.random.rand(pop_size, d)
        mu = np.random.rand(pop_size, d)
        deta = np.minimum(pop_crossover - self.x_lb, self.x_ub - pop_crossover) / (self.x_ub - self.x_lb)
        detaq = np.zeros((pop_size, d))
        # use dot multiply to replace matrix & in matlab
        position1 = (rand_var <= pro_m) * (mu <= 0.5)
        position2 = (rand_var <= pro_m) * (mu > 0.5)
        tmp1 = 2 * mu[position1] + (1 - 2 * mu[position1]) * power(1 - deta[position1], (dis_m + 1))
        detaq[position1] = power(tmp1, 1 / (dis_m + 1)) - 1
        tmp2 = 2 * (1 - mu[position2]) + 2 * (mu[position2] - 0.5) * power(1 - deta[position2], (dis_m + 1))
        detaq[position2] = 1 - power(tmp2, 1 / (dis_m + 1))
        pop_mutation = pop_crossover + detaq * (self.x_ub - self.x_lb)
        return pop_mutation

def init_samples(dim, lb, ub, n_samples):
    return lhs(dim, samples=n_samples) * (ub - lb) + lb
