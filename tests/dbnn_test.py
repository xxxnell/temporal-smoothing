import unittest
import tensorflow as tf
import time
from dbnn import DBNN
from och import OCH


class TestDBNN(unittest.TestCase):

    def test_update(self):
        sample_no = 10
        x_dims = [10, 100]
        y_dims = [10]
        op = lambda x: [tf.tensordot(tf.reshape(x[1], [10, 10]), x[0], 1)]
        och_x1_params = {'k': 5, 'l': 5.0, 's': 1.0}
        och_x_params = {'k': 5, 'l': 5.0, 's': 1.0}
        och_y_params = {'k': 5, 'l': 5.0, 's': 1.0}
        x_0s = [tf.random.normal(shape=[x_dims[0]]) for _ in range(sample_no)]
        x_1s = [[tf.random.normal(shape=[dim]) for dim in x_dims[1:]] for _ in range(och_x1_params['k'])]

        och_x_1 = OCH(**och_x1_params, dims=x_dims[1:], hash_no=3, cs=x_1s)
        och_x = OCH(**och_x_params, dims=x_dims, hash_no=3)
        och_y = OCH(**och_y_params, dims=y_dims, hash_no=3)
        dbnn = DBNN(op, och_x_1, och_x, och_y)

        for x_0 in x_0s:
            dbnn.update(x_0)

        self.assertFalse(och_x.is_empty())
        self.assertFalse(och_y.is_empty())


if __name__ == '__main__':
    unittest.main()

