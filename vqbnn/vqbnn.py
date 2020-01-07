class VQBNN:
    """
    DBNN for data stream, where DNN is augmented with OCHs as distribution estimator units of input and output stream.
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

    def clear(self):
        self.och_x.clear()
        self.och_y.clear()
        self.table.clear()

    def update(self, x_0, n=1.0):
        # Step A. Update OCH_X
        x_1 = self.och_x_1.sample()
        x = [x_0] + x_1 if x_1 is not None else [x_0]
        c_new, n_diff, c_olds = self.och_x.update(x, n)

        # Step B. Execute NN
        if c_new is not None:
            y = self.op(x)
            self.table.append((x, y))
        self.table = [(c, y) for c, y in self.table if len([c_old for c_old in c_olds if c_old is c]) is 0]

        # Step C. Update OCH_Y
        c = self.och_x.search(x_0, 0)
        if c is not None:
            y = next(iter([y for x, y in self.table if c is x]))
            self.och_y.update(y, n_diff / self.och_x.n_tot() if n_diff > 0.0 else 0.0)
