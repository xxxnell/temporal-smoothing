import math
import numpy as np
from nns import ANN
import tensorflow as tf
import time

class OCH():
    """
    Online Codevector Histogram (OCH)

    :param cns: (codevector, count) tuple list
    """

    def __init__(self, k, l, s, dims, hash_no, cns=None, w=None, cache_no=None):
        """
        :param k: hyperparameter K
        :param l: hyperparameter λ
        :param s: hyperparameter σ(φ)
        """
        if cns is None:
            cns = []
        if cache_no is None:
            cache_no = [10] * len(dims)

        self.k = k
        self.l = l
        self.s = s
        self.cns = []
        self.nns = ANN.every(dims, hash_no, w=w, cache_no=cache_no)

        self.phi = self._logit(self.s)
        self.pi_avg = 1 / self.k

        for c, n in cns:
            self.add(c, n)

    def add(self, x, n):
        self.cns.append((x, n))
        self.nns[0].add(x)
        for nns, x_i in zip(self.nns[1:], x):
            nns.add(x_i)

    def remove(self, x):
        self.cns = [(c, n) for c, n in self.cns if c != x]
        self.nns[0].remove(x)
        for nns, c_i in zip(self.nns[1:], x):
            nns.remove(c_i)

    def update(self, x, n=1.0):
        """
        :param x: input vector
        :param n: count of input vector
        :return: new codevector if exists, difference of the count (δn)
        """
        c_new, n_diff = None, n

        c = self.nns[0].search(x)
        if c is None:
            c, m = x, n
            self.add(x, n)
        else:
            # Step A. Increase the count
            m = next(iter([_n for _c, _n in self.cns if _c == c]))
            self.cns = [(_c, _n + n) if _c == c else (_c, _n) for _c, _n in self.cns]

            # Step B. Add a new codeword vector
            n_tot = self.n_tot()
            gamma = math.exp(-1 * self.l / self.n_tot())
            if self._bernoulli(self._prob_add(m / n_tot)) is 1:
                c_new, n_diff = x, gamma * m
                self.add(x, gamma * m)
                self.cns = [(_c, (1 - gamma) * _n) if _c == c else (_c, _n) for _c, _n in self.cns]
            self.cns = [(_c, _n * gamma) for _c, _n in self.cns]

            # Step C. Remove old codeword vectors
            for _c, _n in self.cns:
                n_tot = self.n_tot()
                if self._bernoulli(self._prob_remove(_n / n_tot)) is 1:
                    self.remove(_c)

        return c_new, n_diff

    @staticmethod
    def _bernoulli(p):
        return next(iter(np.random.binomial(1, p, 1))).item()

    @staticmethod
    def _categorical(ps):
        return next(iter(np.where(np.random.multinomial(1, ps) == 1)[0]), None)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _logit(x):
        if x >= 1:
            return math.inf
        elif x <= -1:
            return -math.inf
        else:
            return np.log(x / (1 - x))

    def _prob_add(self, pi):
        return self._sigmoid(pi - self.pi_avg + self.phi)

    def _prob_remove(self, pi):
        return self._sigmoid(self.pi_avg - pi + self.phi) * self.pi_avg

    def n_tot(self):
        return sum([n for _, n in self.cns])

    def sample(self):
        n_tot = self.n_tot()
        c, _ = self.cns[self._categorical([_n / n_tot for _, _n in self.cns])]
        return c

    def search(self, x, i=None):
        if i is None:
            return self.nns[0].search(x)
        else:
            return self.nns[i + 1].search(x)

    def expected(self):
        n_tot = self.n_tot()
        svs = [[c_i * (n / n_tot) for c_i in c] for c, n in self.cns]
        return [sum(vs) for vs in list(map(list, zip(*svs)))]

