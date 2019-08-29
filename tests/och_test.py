import unittest
import tensorflow as tf
import time
from och import OCH


class TestOCH(unittest.TestCase):

    @unittest.skip
    def test_op(self):
        sample_no = 20
        k, l, s, hash_no = 5, 1, 1, 5
        dims = [1, 10 ** 7]
        och = OCH(k, l, s, dims, hash_no, cache_no=[10, 10])
        x1 = tf.ones(shape=[dims[1]])
        xs = [[tf.random.normal(shape=[dims[0]]), x1] for _ in range(sample_no)]

        dts = []
        for x in xs:
            time1 = time.time()
            och.update(x)
            time2 = time.time()
            dts.append(time2 - time1)

        for i, dt in enumerate(dts):
            print("Runtime of #%d execution: " % i, dt * 1000, "ms")
        print("Initial runtime:", dts[0] * 1000, "ms", ", Average runtime: ", sum(dts) / len(dts) * 1000, "ms")

    def test_init_1(self):
        k, l, s, dims, hash_no = 5, 1, 1, [1, 2, 3], 5
        och = OCH(k, l, s, dims, hash_no)

        self.assertEqual(len(och.cns), 0)
        self.assertEqual(len(och.nns), len(dims) + 1)
        self.assertEqual([nns.is_empty() for nns in och.nns], [True] * (len(dims) + 1))

    def test_init_2(self):
        k, l, s, dims, hash_no = 5, 1, 1, [1, 2, 3], 5
        cns = [([tf.ones(shape=[dim]) for dim in dims], 1.0) for _ in range(10)]
        och = OCH(k, l, s, dims, hash_no, cns)

        self.assertEqual(len(och.cns), len(cns))
        self.assertEqual(len(och.nns), len(dims) + 1)
        self.assertEqual([nns.is_empty() for nns in och.nns], [False] * (len(dims) + 1))

    def test_update_1(self):
        warmup = 50
        k, l, s, dims, hash_no = 5, 1.0, 0.5, [1, 2, 3], 5
        och = OCH(k, l, s, dims, hash_no)
        xs = [[tf.random.normal(shape=[dim]) for dim in dims] for _ in range(warmup)]

        for x in xs:
            och.update(x)
            self.assertTrue(len(och.cns) < k * 3)

    def test_update_2(self):
        warmup = 50
        k, l, s, dims, hash_no = 5, 1.0, 1.0, [1, 2, 3], 5
        och = OCH(k, l, s, dims, hash_no)
        xs = [[tf.random.normal(shape=[dim]) for dim in dims] for _ in range(warmup)]

        for x in xs:
            c_new, n_diff, _ = och.update(x)
            self.assertEqual(c_new, x)
            self.assertTrue(n_diff > 0)

    def test_sample(self):
        warmup = 50
        k, l, s, dims, hash_no = 5, 1.0, 0.5, [1, 2, 3], 5
        och = OCH(k, l, s, dims, hash_no)
        xs = [[tf.random.normal(shape=[dim]) for dim in dims] for _ in range(warmup)]
        for x in xs:
            och.update(x)

        self.assertTrue(och.sample() is not None)

    def test_search_1(self):
        warmup = 50
        k, l, s, dims, hash_no = 5, 1.0, 0.5, [1, 2, 3], 5
        och = OCH(k, l, s, dims, hash_no)
        xs = [[tf.random.normal(shape=[dim]) for dim in dims] for _ in range(warmup)]
        zero = [tf.zeros(shape=[dim]) for dim in dims]
        for x in xs:
            och.update(x)

        self.assertNotEqual(och.search(zero), None)
        self.assertEqual([tf.shape(c_i).numpy() for c_i in och.search(zero)], dims)

    def test_search_2(self):
        warmup = 50
        k, l, s, dims, hash_no = 5, 1.0, 0.5, [1, 2, 3], 5
        och = OCH(k, l, s, dims, hash_no)
        xs = [[tf.random.normal(shape=[dim]) for dim in dims] for _ in range(warmup)]
        zero = [tf.zeros(shape=[dim]) for dim in dims]
        for x in xs:
            och.update(x)

        self.assertNotEqual([och.search(zero[i], i) for i in range(len(dims))], [None] * len(dims))
        self.assertEqual([[len(c_i) for c_i in och.search(zero[i], i)] for i in range(len(dims))], [dims] * len(dims))

    def test_expected(self):
        warmup = 50
        k, l, s, dims, hash_no = 30, 1.0, 0.5, [1, 2, 3], 5
        och = OCH(k, l, s, dims, hash_no)
        xs = [[tf.random.normal(shape=[dim]) for dim in dims] for _ in range(warmup)]
        for x in xs:
            och.update(x)

        expected = och.expected()
        norm = tf.math.sqrt(sum([tf.math.square(tf.norm(expected_i)) for expected_i in expected]))

        self.assertTrue(norm < 2.0, "The norm of the expected value {} is too large.".format(norm))
        self.assertEqual([tf.shape(expected_i).numpy() for expected_i in expected], dims)


if __name__ == '__main__':
    unittest.main()

