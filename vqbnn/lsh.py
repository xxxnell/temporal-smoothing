import math
import numpy as np
import tensorflow as tf


class LSH:
    """
    Stable distribution Locality-sensitive hashing (LSH) for input vector.
    h(x) = ⌊ (a * x + b) / w ⌋, where a ~ N(0, 1) and b ~ U[0, w] for given w.

    cache: ([(x_0, ax_0), ...], [(x_1, ax_1), ...], ...)

    See https://en.wikipedia.org/wiki/Locality-sensitive_hashing#Stable_distributions or
        http://mlwiki.org/index.php/Euclidean_LSH
    """

    def __init__(self, dims, w, cache_no=None):
        if cache_no is None:
            cache_no = [0] * len(dims)

        a = [self._normal(dim) for dim in dims]
        b = self._uniform(w)

        self.w = w
        self.a = a
        self.b = b
        self.cache = tuple([[] for _ in range(len(dims))])
        self.cache_no = cache_no

    def clear(self):
        for cache_i in self.cache:
            cache_i.clear()

    @staticmethod
    def _normal(dim):
        return tf.random.normal(shape=[dim])

    @staticmethod
    def _uniform(w):
        return np.random.uniform(high=w)

    def hash(self, x, i=None):
        if i is None:
            ax = sum([self._ax(x_i, i) for x_i, i in zip(x, range(len(self.a)))])
        else:
            ax = self._ax(x, i)
        return math.ceil((ax + self.b) / self.w)

    def _ax(self, x, i):
        ax = next(iter([ax_i for x_i, ax_i in self.cache[i] if x is x_i]), None)
        if ax is None:
            ax = tf.tensordot(self.a[i], tf.cast(tf.reshape(x, [-1]), tf.float32), 1)
            self.cache[i].append((x, ax))
            if len(self.cache[i]) > self.cache_no[i]:
                self.cache[i].pop()
        return ax
