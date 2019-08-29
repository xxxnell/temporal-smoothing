from och import OCH
import tensorflow as tf


class DBNN:
    """
    DBNN for data stream.
    """

    def __init__(self, op, och_x_1, och_x, och_y, table=None):
        """
        :param op: NN([x_0, x_1, ...]). Note that return type must be [y_0, ...].
        :param och_x_1: posterior distribution
        :param och_x: input vector distribution
        :param och_y: output vector distribution
        :param table: lookup table from x to y
        """
        if table is None:
            table = []

        self.op = op
        self.och_x_1 = och_x_1
        self.och_x = och_x
        self.och_y = och_y
        self.table = table

    def update(self, x_0, n=1.0):
        x_1 = self.och_x_1.sample()
        x = [x_0] + x_1
        c_new, n_diff, c_olds = self.och_x.update(x, n)
        if c_new is not None:
            y = self.op(x)
            self.table.append((x, y))
        self.table = [(c, y) for c, y in self.table if c not in c_olds]
        c = self.och_x.search(x_0, 0)
        y = next(iter([y for x, y in self.table if c is x]))
        self.och_y.update(y, n_diff / self.och_x.n_tot())
