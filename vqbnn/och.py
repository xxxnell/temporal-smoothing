import math
import numpy as np
from vqbnn.nns import ANN
import tensorflow as tf


class OCH():
    """
    Online Codevector Histogram (OCH) to estimate the probability of non-stationary data stream as well as dataset.
    """

    def __init__(self, k, l, s, dims, hash_no, cns=None, cs=None, w=None, cache_no=None):
        """
        :param k: hyperparameter K
        :param l: hyperparameter λ
        :param s: hyperparameter σ(φ)
        """
        if cs is not None and cns is None:
            cns = [(c, 1.0) for c in cs]
        if cns is None:
            cns = []
        if cache_no is None:
            cache_no = [10] * len(dims)

        self.k = k
        self.l = l
        self.s = s
        self.dims = dims
        self.cns = []
        self.nns = ANN.every(dims, hash_no, w=w, cache_no=cache_no)

        self.phi = self._logit(self.s)
        self.pi_avg = 1 / self.k

        for c, n in cns:
            self.add(c, n)

    def clear(self):
        self.cns.clear()
        for nns_i in self.nns:
            nns_i.clear()

    def add(self, x, n):
        self.cns.append((x, n))
        self.nns[0].add(x)
        for nns, x_i in zip(self.nns[1:], x):
            nns.add(x_i)

    def remove(self, x):
        self.cns = [(c, n) for c, n in self.cns if c is not x]
        self.nns[0].remove(x)
        for nns, c_i in zip(self.nns[1:], x):
            nns.remove(c_i)

    def update(self, x, n=1.0):
        """
        :param x: input vector
        :param n: count of input vector
        :return: new codevector if exists, difference of the count (δn)
        """
        c_new, n_diff, c_olds = None, 0.0, []

        c = self.nns[0].search(x)
        exists = len([_c for _c, _ in self.cns if c is _c]) > 0
        if c is None or not exists:
            c_new, n_diff = x, n
            self.add(x, n)
        else:
            # Step A. Increase the count
            n_diff = n
            m = next(iter([_n for _c, _n in self.cns if _c is c])) + n
            self.cns = [(_c, m) if _c is c else (_c, _n) for _c, _n in self.cns]

            # Step B. Add a new codeword vector
            n_tot = self.n_tot()
            gamma = math.exp(-1 * self.l / n_tot) if n_tot > 0.0 else 0.0
            if self._bernoulli(self._prob_add(m / n_tot if n_tot > 0.0 else 1.0)) is 1:
                c_new, n_diff = x, (1 - gamma) * m
                self.cns = [(_c, gamma * m) if _c is c else (_c, _n) for _c, _n in self.cns]
                self.add(x, (1 - gamma) * m)
            self.cns = [(_c, _n * gamma) for _c, _n in self.cns]

            # Step C. Remove old codeword vectors
            n_tot = self.n_tot()
            for _c, _n in self.cns:
                if self._bernoulli(self._prob_remove(_n / n_tot if n_tot > 0.0 else 1.0)) is 1:
                    self.remove(_c)
                    c_olds.append(_c)
                    if c_new is None and _c is c or _c is c_new:
                        c_new, n_diff = None, 0.0

        return c_new, n_diff, c_olds

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
        elif x <= 0:
            return -math.inf
        else:
            return np.log(x / (1 - x))

    def _prob_add(self, pi):
        return self._sigmoid(pi - self.pi_avg + self.phi)

    def _prob_remove(self, pi):
        return self._sigmoid(self.pi_avg - pi + self.phi) * self.pi_avg

    def n_tot(self):
        return sum([n for _, n in self.cns])

    def is_empty(self):
        return len(self.cns) is 0 and self.nns.is_empty()

    def len(self):
        return len(self.cns)

    def cws(self):
        n_tot = self.n_tot()
        return [(c, n / n_tot) if n_tot > 0 else (c, 0.0) for c, n in self.cns]

    def sample(self):
        if self.cws():
            c, _ = self.cns[self._categorical([w for _, w in self.cws()])]
        else:
            c = None
        return c

    def search(self, x, i=None):
        if i is None:
            return self.nns[0].search(x)
        else:
            c_i = self.nns[i + 1].search(x)
            cs = [c for c, _ in self.cns if c[i] is c_i]
            return next(iter(cs), None)

    def moments(self):
        axes = [0]
        if len(self.cws()) > 0:
            cs, ws = zip(*self.cws())

            mean, variance = [], []
            for c_is in zip(*cs):
                ws_extend = [tf.ones(tf.shape(c_i)) * w for w, c_i in zip(ws, c_is)]
                mean_i, variance_i = tf.nn.weighted_moments(tf.stack(c_is), axes, tf.stack(ws_extend))
                mean.append(mean_i)
                variance.append(variance_i)
        else:
            mean, variance = None, None
        return mean, variance

    def expected(self):
        return self.moments()[0]

    def var(self):
        return self.moments()[1]

