# -*- coding: utf-8 -*-
#!/usr/bin/env python

from fd_sketch import *
import numpy as np
import unittest

class TestFrequentDirectionSketch(unittest.TestCase):

    def test_error(self):
        mat_a = np.identity(2)
        mat_b = np.array([[0, 2], [2, 0]])
        self.assertEqual(3.0, calculateError(mat_a, mat_b))

    def test_frobenius_norm(self):
        # Output example of Frobenius Norm for magical matrix of size 3
        # Example taken from http://www.mathworks.co.jp/jp/help/symbolic/norm.htm
        mat_magic_inv = ln.inv(np.array([[2, 9, 4], [7, 5, 3], [6, 1, 8]]))
        self.assertAlmostEqual((391 ** (0.5) / 60) ** 2, squaredFrobeniusNorm(mat_magic_inv))

    def test_too_large_ell(self):
        self.assertRaises(ValueError, sketch, np.random.randn(1000,100), 500)
        self.assertRaises(ValueError, sketch, np.random.randn(10,100), 50)

    def test_rand_matrix(self):
        for ell in [2, 10, 100, 199]:
            mat_a = np.random.randn(1000, 100)
            mat_b = sketch(mat_a, ell)
            print 'error vs upper-bound: ', calculateError(mat_a, mat_b), ' vs ', 2 * squaredFrobeniusNorm(mat_a) / ell
            self.assertGreaterEqual(2 * squaredFrobeniusNorm(mat_a) / ell, calculateError(mat_a, mat_b))

    def test_sparse_rand_matrix(self):
        for ell in [2, 10, 100, 199]:
            mat_a = np.random.randn(1000, 100)
            mat_a[mat_a < 2.0] = 0.0
            mat_b = sketch(mat_a, ell)
            print 'error vs upper-bound: ', calculateError(mat_a, mat_b), ' vs ', 2 * squaredFrobeniusNorm(mat_a) / ell
            self.assertGreaterEqual(2 * squaredFrobeniusNorm(mat_a) / ell, calculateError(mat_a, mat_b))

if __name__ == '__main__':
    unittest.main()
