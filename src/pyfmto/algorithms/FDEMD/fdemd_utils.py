import copy
import math
import numpy as np
from pyDOE import lhs
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from tabulate import tabulate
from pyfmto.utilities import logger


def power(mat1, mat2):
    # ----------------- #
    # Copied from FMTBO #
    # ----------------- #
    # To solve the problem: Numpy does not seem to
    # allow fractional powers of negative numbers
    return np.sign(mat1) * np.power(np.abs(mat1), mat2)


def mini_batches(input_x, input_y, distance, batch_size=64, seed=0):
    """
    return random batch indexes for a list
    """
    np.random.seed(seed)
    num_samp = input_x.shape[0]
    batches = []
    permutation = list(np.random.permutation(num_samp))
    num_batch = math.floor(num_samp / batch_size)
    iter_list = [i for i in range(num_batch)]

    for k in iter_list:
        batch_index = permutation[k * batch_size:(k + 1) * batch_size]
        batches.append((input_x[batch_index], input_y[batch_index], distance[batch_index]))
    if num_samp % batch_size != 0:
        batch_index = permutation[batch_size * num_batch:]
        batches.append((input_x[batch_index], input_y[batch_index], distance[batch_index]))

    return batches


def index_bootstrap(num_data: int, prob: float):
    """

    Parameters
    ----------
    num_data : int
        The index matrix of input
    prob : float
        The probability for one index sample to be chosen(>0)

    Returns
    -------
        bool_index : np.ndarray(bool)
        Index of chosen samples

    Examples
    --------
    >>> index_bootstrap(num_data=5, prob=0.5)
    array([ True, False, False, False,  True])
    """
    rand_p = np.random.rand(num_data)
    out = np.greater(rand_p, 1 - prob)
    if True not in out:
        out = index_bootstrap(num_data, prob)
    return out


class AggData:
    def __init__(self, version, src_num, agg_res):
        self.version = version
        self.src_num = src_num
        self.agg_res = agg_res


class RadialBasisFunctionNetwork:
    def __init__(self, dim, obj, kernel_size, optimizer='sgd', epoch=5, lr=0.06, alpha=1.0):
        # network args
        self._dim = dim
        self._obj = obj
        self._kernel_size = kernel_size
        self._optimizer = optimizer
        self._epoch = epoch
        self._lr = lr
        self._alpha = alpha

        # model params
        self._weight = None
        self._bias = None
        self._centers = None
        self._std = None

    def sync_auto(self, params):
        orig_keys = tuple(self.__dict__.keys())
        self.__dict__.update(params)
        curr_keys = tuple(self.__dict__.keys())
        if set(orig_keys) != set(curr_keys):
            logger.warning(f'model params may be not synchronized correctly\n'
                            f'original keys: {orig_keys}\nnew keys: {set(curr_keys)-set(orig_keys)}')

    def sync_manuel(self, weight=None, bias=None, centers=None, std=None):
        """
        Overwrite the given non-none params.

        Parameters
        ----------
        weight
        bias
        centers
        std

        Returns
        -------

        """
        if weight is not None:
            self._weight = weight
        if bias is not None:
            self._bias = bias
        if centers is not None:
            self._centers = centers
        if std is not None:
            self._std = std

    def train(self, x, y):
        if not self.trainable:
            raise RuntimeError(f'Error: model is not trainable\n{self.train_requirements}')
        num_x = x.shape[0]
        self._centers, self._std = self._find_center(x)
        distance = self._dist(self._centers, x.T)
        spreads_mat = np.tile(self._std.reshape(-1, 1), (1, num_x))
        hidden_layer_output = np.exp(-(distance / spreads_mat) ** 2).T

        if self._optimizer == 'sgd':
            self._sgd(x, y, hidden_layer_output)
        elif self._optimizer == 'max-gd':
            self._max_gd(y, hidden_layer_output)
        elif self._optimizer == 'm-sgd':
            self._mini_batch_sgd(x, y, hidden_layer_output)
        elif self._optimizer == '1-sgd':
            self._one_sgd(x, y, hidden_layer_output)
        else:
            raise ValueError('Error: optimizer name')

    def _find_center(self, train_x):
        k_means = KMeans(n_clusters=self._kernel_size)
        k_means.fit(train_x)
        centers = k_means.cluster_centers_
        centers, tmp_index = self._sort_centers(centers)
        all_distances = self._dist(centers, centers.T)
        d_max = all_distances.max(initial=-10)

        for i in range(self._kernel_size):
            all_distances[i, i] = d_max + 1
        all_distances = np.where(all_distances != 0, all_distances, 0.000001)
        std = self._alpha * np.min(all_distances, axis=0)
        return centers, std

    def predict(self, x):
        if not self.predictable:
            raise RuntimeError('Error: model is not predictable')
        num_x = x.shape[0]
        center_to_test_distances = self._dist(self._centers, x.T)
        spread_matrix = np.tile(self._std.reshape(-1, 1), (1, num_x))
        hidden_layer_output = np.exp(-(center_to_test_distances / spread_matrix) ** 2).T
        predicted_output = hidden_layer_output @ self._weight + self._bias
        return predicted_output

    def _sgd(self, x, y, hidden_layer_output):
        samp_id = [i for i in range(x.shape[0])]
        for _ in range(self._epoch):
            for i in samp_id:
                F = hidden_layer_output[i].T @ self._weight + self._bias
                error = (F - y[i]).flatten()
                dw = hidden_layer_output[i].reshape(-1, 1) * error.reshape(1, self._obj)
                db = error
                # update
                self._weight = self._weight - self._lr * dw
                self._bias = self._bias - self._lr * db

    def _max_gd(self, train_y, hidden_layer_output):
        for _ in range(self._epoch):
            F = hidden_layer_output @ self._weight + self._bias
            error = F - train_y
            sum_error = np.sum(error ** 2, axis=1)
            max_index = np.argmax(sum_error)
            dw = hidden_layer_output[max_index].reshape(-1, 1) * error[max_index].reshape(1, self._obj)
            db = error[max_index].reshape(1, self._obj)

            self._weight = self._weight - self._lr * dw
            self._bias = self._bias - self._lr * db

    def _mini_batch_sgd(self, train_x, train_y, hidden_layer_output):
        batch_size = 12
        for _ in range(self._epoch):

            batches = mini_batches(train_x, train_y, hidden_layer_output, batch_size, 1234)

            for batch_ in batches:
                batch_x, batch_y, hidden_layer_output_tmp = batch_[0], batch_[1], batch_[2]
                real_bs = batch_x.shape[0]
                F = hidden_layer_output_tmp @ self._weight + self._bias
                error = F - batch_y
                tmp_hidden_layer_output = hidden_layer_output_tmp.reshape(real_bs, -1, 1)
                tmp_error = error.reshape(real_bs, -1, self._obj)
                out = tmp_hidden_layer_output * tmp_error
                dw = np.sum(out, axis=0) / real_bs
                db = np.sum(error, axis=0).reshape(1, self._obj) / real_bs

                self._weight = self._weight - self._lr * dw
                self._bias = self._bias - self._lr * db

    def _one_sgd(self, train_x, train_y, hidden_layer_output):
        wanted_size = 32
        num_x = train_x.shape[0]
        prob = wanted_size / num_x
        for _ in range(self._epoch):
            batch_index = index_bootstrap(num_x, prob)
            batch_x, batch_y = train_x[batch_index], train_y[batch_index]
            batch_size = batch_x.shape[0]
            hidden_layer_output_tmp = hidden_layer_output[batch_index]

            F = hidden_layer_output_tmp @ self._weight + self._bias
            error = F - batch_y
            tmp_hidden_layer_output = hidden_layer_output_tmp.reshape(batch_size, -1, 1)
            tmp_error = error.reshape(batch_size, -1, self._obj)
            out = tmp_hidden_layer_output * tmp_error
            dw = np.sum(out, axis=0) / batch_size
            db = np.sum(error, axis=0).reshape(1, self._obj) / batch_size

            self._weight = self._weight - self._lr * dw
            self._bias = self._bias - self._lr * db

    @staticmethod
    def _sort_centers(centers):
        """
        To sort the centers according to the distance from zero vector
        Please note that this fun has not considered the direction of the centers, should be change
        :param centers:
        :return: sorted centers & index
        """
        tmp_centers = copy.deepcopy(centers)
        distance = np.sum(tmp_centers ** 2, axis=1)
        sorted_index = np.argsort(distance)
        tmp_centers = tmp_centers[sorted_index, :]
        return tmp_centers, sorted_index

    @staticmethod
    def _dist(mat1, mat2):
        """
        rewrite euclidean distance function in Matlab: dist
        :param mat1: matrix 1, M x N
        :param mat2: matrix 2, N x R
        output: Mat3. M x R
        """
        mat2 = mat2.T
        return cdist(mat1, mat2)

    @property
    def params(self):
        all_params = {
            '_weight': self._weight,
            '_bias': self._bias,
            '_centers': self._centers,
            '_std': self._std
        }
        return {k: v for k, v in all_params.items() if v is not None}

    @property
    def trainable(self):
        has_value = [self._epoch is not None, self._lr is not None, self._alpha is not None,
                     self._weight is not None, self._bias is not None, self._optimizer is not None]
        return all(has_value)

    @property
    def predictable(self):
        has_value = [self._weight is not None, self._bias is not None,
                     self._centers is not None, self._std is not None]
        return all(has_value)

    @property
    def train_requirements(self):
        headers = ['epoch', 'lr', 'alpha', 'weight', 'bias', 'optimizer']
        values = [[self._epoch, self._lr, self._alpha, self._weight, self._bias, self._optimizer]]
        indices = ['value']
        tab = tabulate(values, headers, showindex=indices, missingval='-', tablefmt='simple_grid')
        return tab

    @property
    def predict_requirements(self):
        headers = ['weight', 'bias', 'centers', 'std']
        values = [[self._weight, self._bias, self._centers, self._std]]
        indices = ['value']
        tab = tabulate(values, headers, showindex=indices, missingval='-', tablefmt='simple_grid')
        return tab

    @property
    def wb_str(self):
        w_str = np.array2string(self._weight.flatten(), formatter={'float_kind': lambda x: f"{x:.2e}"})
        b_str = np.array2string(self._bias.flatten(), formatter={'float_kind': lambda x: f"{x:.2e}"})
        return f"w({w_str}) b({b_str})"


class GeneticAlgorithm:

    def __init__(self):
        pass

    @staticmethod
    def _initialize_pop(n, d, lb, ub):
        result = lhs(d, samples=n)
        return result * (ub - lb) + lb

    @staticmethod
    def _obj_wrapper(func, args, kwargs, x):
        return func(x, *args, **kwargs)

    def RCGA(self, obj_func, lb, ub, pop_size=100, max_iter=100, particle_output=False):
        lb = np.array(lb)
        ub = np.array(ub)

        d = len(lb)

        pop_rand = self._initialize_pop(pop_size, d, lb[0], ub[0])

        pop_fitness, _ = obj_func(pop_rand)

        generation = 0

        while generation < max_iter:
            temp1 = np.random.randint(0, pop_size, pop_size)
            temp2 = np.random.randint(0, pop_size, pop_size)

            pop_parent = np.zeros((pop_size, d))
            for i in range(pop_size):
                if pop_fitness[temp1[i]] <= pop_fitness[temp2[i]]:
                    pop_parent[i] = pop_rand[temp1[i]]
                else:
                    pop_parent[i] = pop_rand[temp2[i]]

            # crossover(simulated binary crossover)
            # dic_c is the distribution index of crossover
            dis_c = 1
            mu = np.random.rand(int(pop_size / 2), d)
            idx1 = [i for i in range(0, pop_size, 2)]
            idx2 = [i + 1 for i in range(0, pop_size, 2)]
            parent1 = pop_parent[idx1, :]
            parent2 = pop_parent[idx2, :]
            element_min = np.minimum(parent1, parent2)
            element_max = np.maximum(parent1, parent2)
            tmp_min = np.minimum(element_min - lb, ub - element_max)
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

            # mutation (ploynomial mutation)
            # dis_m is the distribution index of polynomial mutation
            dis_m = 1
            pro_m = 1 / d
            rand_var = np.random.rand(pop_size, d)
            mu = np.random.rand(pop_size, d)
            deta = np.minimum(pop_crossover - lb, ub - pop_crossover) / (ub - lb)
            detaq = np.zeros((pop_size, d))
            # use dot multiply to replace matrix & in matlab
            position1 = (rand_var <= pro_m) * (mu <= 0.5)
            position2 = (rand_var <= pro_m) * (mu > 0.5)
            tmp1 = 2 * mu[position1] + (1 - 2 * mu[position1]) * power(1 - deta[position1], (dis_m + 1))
            detaq[position1] = power(tmp1, 1 / (dis_m + 1)) - 1
            tmp2 = 2 * (1 - mu[position2]) + 2 * (mu[position2] - 0.5) * power(1 - deta[position2], (dis_m + 1))
            detaq[position2] = 1 - power(tmp2, 1 / (dis_m + 1))
            pop_mutation = pop_crossover + detaq * (ub - lb)

            # fitness calculation
            pop_mutation_fitness, _ = obj_func(pop_mutation)
            # ------------------------------------ environment selection
            pop_rand_iter = np.vstack((pop_rand, pop_mutation))
            pop_fitness_iter = np.concatenate((pop_fitness, pop_mutation_fitness))
            sorted_fit_index = np.argsort(pop_fitness_iter)
            pop_rand = pop_rand_iter[sorted_fit_index[0:pop_size], :]
            pop_fitness = pop_fitness_iter[sorted_fit_index[0:pop_size]]
            generation += 1
            # print(pop_fitness)

        temp_best = pop_rand[0, :].flatten()
        _, pop_uncertainty = obj_func(pop_rand)
        sorted_unc_index = np.argsort(pop_uncertainty)
        p_min = pop_rand[sorted_unc_index[0], :]
        if particle_output:
            # return p_min, temp_best, pop_rand, pop_fitness, pop_uncertainty
            return p_min, temp_best, pop_rand, pop_uncertainty
        else:
            return p_min, temp_best


if __name__ == '__main__':
    model1 = RadialBasisFunctionNetwork(dim=2, obj=1, kernel_size=5)
    model2 = RadialBasisFunctionNetwork(dim=2, obj=2, kernel_size=5)
    print(model1.trainable)
    # params2 = model2.params
    # params2.update({'a': 0})
    # model1.sync_auto(params2)