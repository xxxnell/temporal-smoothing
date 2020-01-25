import numpy as np
import tensorflow as tf
from vqbnn.lsh import LSH


class ANN:
    """
    Approximate nearest neighbor search for concat input vector (i.e., x = x0 ⊕ x1 ⊕ ...) using LSH.
    """

    def __init__(self, lshs, i=None, xhashs=None, hashxs=None):
        """
        :param lshs: set of LSHs
        :param i: index of x (i.e., x_i)
        :param xhashs: [(x, (hash1, hash2, ...))]
        :param hashxs: ({hash1: [x], ...}, {hash2: [x], ...})
        """
        if xhashs is None:
            xhashs = []
        if hashxs is None:
            hashxs = tuple([{} for _ in range(len(lshs))])

        self.lshs = lshs
        self.xhashs = xhashs
        self.hashxs = hashxs
        self.i = i

    @classmethod
    def every(cls, dims, hash_no, w=None, cache_no=None):
        if w is None:
            w = [0.5] * hash_no

        lshs = []
        for j in range(hash_no):
            lshs.append(LSH(dims, w[j], cache_no))

        anns = [cls(lshs)]
        for i in range(len(dims)):
            anns.append(cls(lshs, i))

        return anns

    def clear(self):
        for lsh in self.lshs:
            lsh.clear()
        self.xhashs.clear()
        for hashx in self.hashxs:
            hashx.clear()

    def search(self, x):
        hashs = tuple([lsh.hash(x, self.i) for lsh in self.lshs])
        candidates = []
        for hashx, hash in zip(self.hashxs, hashs):
            if hash in hashx:
                candidates += hashx[hash]

        counts = []
        for x, _ in self.xhashs:
            counts.append(len([x is candidate for candidate in candidates]))
        if counts:
            index, = np.where(counts == np.max(counts))
            nn, _ = self.xhashs[index[-1]]
        else:
            nn = None

        return nn

    @staticmethod
    def full_search(x, cs):
        normsqrs = []
        for c in cs:
            if isinstance(x, list):
                normsqr = sum([tf.math.square(tf.norm(c_i - x_i)) if c_i is not x_i else 0 for c_i, x_i in zip(c, x)])
            else:
                normsqr = tf.math.square(tf.norm(c - x)) if c is not x else 0
            normsqrs.append(normsqr)

        return cs[tf.math.argmin(normsqrs)]

    def add(self, x):
        hashs = tuple([lsh.hash(x, self.i) for lsh in self.lshs])

        self.xhashs.append((x, hashs))

        for hash, hashx in zip(hashs, self.hashxs):
            if hash in hashx:
                hashx[hash].append(x)
            else:
                hashx[hash] = [x]

    def remove(self, x):
        hashs, xhashs = (), []
        for _x, _hashs in self.xhashs:
            if _x is x:
                hashs = _hashs
            else:
                xhashs.append((_x, _hashs))
        self.xhashs = xhashs

        for hash, hashx in zip(hashs, self.hashxs):
            hashx[hash] = [_x for _x in hashx[hash] if x is not _x]
            if len(hashx[hash]) is 0:
                hashx.pop(hash, None)

    def is_empty(self):
        return len(self.xhashs) == 0 and all([len(hashx_dict) == 0 for hashx_dict in self.hashxs])


