# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import numpy.linalg as ln
import math
import sys

""" This is a simple and deterministic method for matrix sketch.
The original method has been introduced in [Liberty2013]_ .

[Liberty2013] Edo Liberty, "Simple and Deterministic Matrix Sketching", ACM SIGKDD, 2013.
"""

def sketch(mat_a, ell):
    """Compute a sketch matrix of input matrix 
    Note that \ell must be smaller than m * 2
    
    :param mat_a: original matrix to be sketched (n x m)
    :param ell: the number of rows in sketch matrix
    :returns: sketch matrix (\ell x m)
    """

    # number of columns
    m = mat_a.shape[1]

    # Input error handling
    if math.floor(ell / 2) >= m:
        raise ValueError('Error: ell must be smaller than m * 2')
    if ell >= mat_a.shape[0]:
        raise ValueError('Error: ell must not be greater than n')

    # initialize output matrix B
    mat_b = np.zeros([ell, m])

    # compute zero valued row list
    zero_rows = np.nonzero([round(s, 7) == 0.0 for s in np.sum(mat_b, axis = 1)])[0].tolist()
    
    # repeat inserting each row of matrix A
    for i in range(0, mat_a.shape[0]):

        # insert a row into matrix B
        mat_b[zero_rows[0], :] = mat_a[i, :]

        # remove zero valued row from the list
        zero_rows.remove(zero_rows[0])

        # if there is no more zero valued row
        if len(zero_rows) == 0:

            # compute SVD of matrix B
            mat_u, vec_sigma, mat_v = ln.svd(mat_b, full_matrices=False)

            # obtain squared singular value for threshold
            squared_sv_center = vec_sigma[math.floor(ell / 2)] ** 2

            # update sigma to shrink the row norms
            sigma_tilda = [(0.0 if d < 0.0 else math.sqrt(d)) for d in (vec_sigma ** 2 - squared_sv_center)]

            # update matrix B where at least half rows are all zero
            mat_b = np.dot(np.diagflat(sigma_tilda), mat_v)

            # update the zero valued row list
            zero_rows = np.nonzero([round(s, 7) == 0 for s in np.sum(mat_b, axis = 1)])[0].tolist()

    return mat_b


def calculateError(mat_a, mat_b):
    """Compute the degree of error by sketching

    :param mat_a: original matrix
    :param mat_b: sketch matrix
    :returns: reconstruction error
    """
    dot_mat_a = np.dot(mat_a.T, mat_a)
    dot_mat_b = np.dot(mat_b.T, mat_b)
    return ln.norm(dot_mat_a - dot_mat_b, ord = 2)


def squaredFrobeniusNorm(mat_a):
    """Compute the squared Frobenius norm of a matrix

    :param mat_a: original matrix
    :returns: squared Frobenius norm
    """
    return ln.norm(mat_a, ord = 'fro') ** 2
