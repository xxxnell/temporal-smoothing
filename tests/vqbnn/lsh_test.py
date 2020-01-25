import unittest
import tensorflow as tf
import time
from vqbnn.lsh import LSH


class TestLSH(unittest.TestCase):

    @unittest.skip
    def test_op(self):
        sample_no = 10
        dims, w, cache_no = [1, 10 ** 7], 3, [10, 10]
        lsh = LSH(dims=dims, w=w, cache_no=cache_no)
        x1 = tf.ones(shape=[dims[1]])
        xs = [[tf.random.normal(shape=[dims[0]]), x1] for _ in range(sample_no)]

        dts = []
        for x in xs:
            time1 = time.time()
            lsh.hash(x)
            time2 = time.time()
            dts.append(time2 - time1)

        for i, dt in enumerate(dts):
            print("Runtime of #%d execution: " %i, dt * 1000, "ms")

    def test_init_1(self):
        dims, w = [1, 2, 3], 10
        lsh = LSH(dims=dims, w=w)
        self.assertEqual([tf.shape(a_i).numpy() for a_i in lsh.a], [[dim] for dim in dims])
        self.assertTrue(0 <= lsh.b <= w)

    def test_init_2(self):
        dims, w = [1, 2, 3], 10
        lsh1 = LSH(dims=dims, w=w)
        lsh2 = LSH(dims=dims, w=w)

        self.assertNotEqual(lsh1.a, lsh2.a)
        self.assertNotEqual(lsh1.b, lsh2.b)

    def test_hash(self):
        dims = [1, 2, 3]
        lsh = LSH(dims=dims, w=10)
        x = [tf.random.normal(shape=[dim]) for dim in dims]
        self.assertIsInstance(lsh.hash(x), int)

    def test_cache(self):
        sample_no = 10
        dims, w, cache_no = [1, 10 ** 7, 1], 3, [3, 4, 5]
        lsh = LSH(dims=dims, w=w, cache_no=cache_no)
        xs = [[tf.random.normal(shape=[dim]) for dim in dims] for _ in range(sample_no)]

        for x in xs:
            lsh.hash(x)
            self.assertEqual(len(lsh.cache), len(dims))
            self.assertTrue(all([len(cache_i) <= cache_no_i for cache_i, cache_no_i in zip(lsh.cache, cache_no)]))


if __name__ == '__main__':
    unittest.main()
