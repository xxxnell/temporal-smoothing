import unittest
import tensorflow as tf
import time
from dbnn.nns import LSH, ANN


class TestANN(unittest.TestCase):

    @unittest.skip
    def test_op(self):
        sample_no, query_no = 1, 10
        dims, hash_no = [1, 10 ** 7], 3
        anns = ANN.every(dims, hash_no, cache_no=[10] * len(dims))
        ann = anns[0]
        xs = [[tf.random.normal(shape=[dim]) for dim in dims] for _ in range(sample_no)]
        query = [tf.zeros(shape=[dim]) for dim in dims]

        for x in xs:
            ann.add(x)

        dts = []
        for _ in range(query_no):
            time1 = time.time()
            ann.search(query)
            time2 = time.time()
            dts.append(time2 - time1)

        for i, dt in enumerate(dts):
            print("Runtime of #%d execution: " % i, dt * 1000, "ms")

    def test_every(self):
        dims, hash_no = [1, 2, 3], 3
        anns = ANN.every(dims, hash_no)
        lshs = [ann.lshs for ann in anns]

        self.assertEqual(len(anns), len(dims) + 1)
        self.assertTrue(all([lsh1 is lsh2 for lsh1, lsh2 in zip(lshs[1:], lshs[:-1])]))
        self.assertEqual([ann.i for ann in anns], [None] + list(range(0, len(dims))))

    def test_search_0(self):
        dims, hash_no = [1, 2, 3], 5
        anns = ANN.every(dims, hash_no)
        sumvec0 = [tf.ones(shape=[dim]) * 0.00 for dim in dims]
        vecs0 = [sumvec0] + sumvec0

        self.assertEqual([ann.search(vec0) for ann, vec0 in zip(anns, vecs0)], [None] * (len(dims) + 1))

    def test_search_1(self):
        dims, hash_no = [1, 2, 3], 5
        anns = ANN.every(dims, hash_no)
        sumvec0 = [tf.ones(shape=[dim]) * 0.00 for dim in dims]
        sumvec1 = [tf.ones(shape=[dim]) * 0.01 for dim in dims]
        vecs0 = [sumvec0] + sumvec0
        vecs1 = [sumvec1] + sumvec1

        for ann, vec1 in zip(anns, vecs1):
            ann.add(vec1)

        self.assertEqual([ann.search(vec0) for ann, vec0 in zip(anns, vecs0)], vecs1)

    def test_search_2(self):
        dims, hash_no = [1, 2, 3], 5
        anns = ANN.every(dims, hash_no)
        sumvec0 = [tf.ones(shape=[dim]) * 0.00 for dim in dims]
        sumvec1 = [tf.ones(shape=[dim]) * 0.01 for dim in dims]
        sumvec2 = [tf.ones(shape=[dim]) * 0.02 for dim in dims]
        vecs0 = [sumvec0] + sumvec0
        vecs1 = [sumvec1] + sumvec1
        vecs2 = [sumvec2] + sumvec2

        for ann, vec1, vec2 in zip(anns, vecs1, vecs2):
            ann.add(vec1)
            ann.add(vec2)

        self.assertEqual([ann.search(vec0) for ann, vec0 in zip(anns, vecs0)], vecs1)

    def test_full_search_1(self):
        dims = [1, 2, 3]
        sumvec0 = [tf.ones(shape=[dim]) * 0.00 for dim in dims]
        sumvec1 = [tf.ones(shape=[dim]) * 0.01 for dim in dims]
        sumvec2 = [tf.ones(shape=[dim]) * 0.02 for dim in dims]

        self.assertEqual(ANN.full_search(sumvec0, [sumvec1, sumvec2]), sumvec1)

    def test_full_search_2(self):
        dim = 10
        vec0 = tf.ones(shape=[dim]) * 0.00
        vec1 = tf.ones(shape=[dim]) * 0.01
        vec2 = tf.ones(shape=[dim]) * 0.02

        self.assertEqual(ANN.full_search(vec0, [vec1, vec2]), vec1)

    def test_add(self):
        dims, hash_no = [1, 2, 3], 5
        anns = ANN.every(dims, hash_no)
        sumvec = [tf.random.normal(shape=[dim]) for dim in dims]
        vecs = [tf.random.normal(shape=[dim]) for dim in dims]

        anns[0].add(sumvec)
        for ann, vec in zip(anns[1:], vecs):
            ann.add(vec)

        self.assertEqual([len(ann.xhashs) for ann in anns], [1] * (len(dims) + 1))
        self.assertEqual([[len(xhashs[1]) for xhashs in ann.xhashs] for ann in anns], [[hash_no]] * (len(dims) + 1))

        self.assertEqual([len(ann.hashxs) for ann in anns], [hash_no] * (len(dims) + 1))
        self.assertEqual(
            [[[len(value) for _, value in hashxs.items()] for hashxs in ann.hashxs] for ann in anns],
            [[[1]] * hash_no] * (len(dims) + 1))

    def test_remove(self):
        dims, hash_no, seed = [1, 2, 3], 5, 100
        anns = ANN.every(dims, hash_no)
        tf.random.set_seed(seed)
        sumvec = [tf.random.normal(shape=[dim]) for dim in dims]
        vecs = [tf.random.normal(shape=[dim]) for dim in dims]

        anns[0].add(sumvec)
        for ann, vec in zip(anns[1:], vecs):
            ann.add(vec)

        anns[0].remove(sumvec)
        for ann, vec in zip(anns[1:], vecs):
            ann.remove(vec)

        self.assertEqual([ann.xhashs for ann in anns], [[]] * (len(dims) + 1))
        self.assertEqual([ann.hashxs for ann in anns], [tuple([{}] * hash_no)] * (len(dims) + 1))


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
